#!/usr/bin/env python3
"""Reference inference for the six-model Qwen3-TTS CoreML export.

Each request owns a fresh CodeDecoder state. SpeechDecoder is fixed-size;
requests exceeding its frame capacity fail instead of silently truncating audio.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np

LANGUAGES = {"chinese":2055,"english":2050,"german":2053,"italian":2070,
             "portuguese":2071,"spanish":2054,"japanese":2058,"korean":2064,
             "french":2061,"russian":2069}


def step_inputs(embedding, position, capacity):
    if not 0 <= position < capacity:
        raise ValueError(f"Position {position} exceeds cache capacity {capacity}")
    mask = np.full((1, capacity), -1e4, dtype=np.float16)
    mask[:, :position+1] = 0
    update = np.zeros_like(mask); update[:, position] = 1
    return {"input_embeds":np.asarray(embedding, dtype=np.float16).reshape(1,-1,1,1),
            "cache_length":np.array([position], dtype=np.int32),
            "key_padding_mask":mask,"kv_cache_update_mask":update}


def sample(logits, rng, temperature=0.8, top_k=50):
    x = np.asarray(logits, dtype=np.float64).reshape(-1)
    if not np.isfinite(x).all():
        raise ValueError("Non-finite model logits")
    if temperature <= 0:
        return int(x.argmax())
    indices = np.argpartition(x, -top_k)[-top_k:]
    p = x[indices] / temperature; p = np.exp(p - p.max()); p /= p.sum()
    return int(rng.choice(indices, p=p))


class Pipeline:
    def __init__(self, directory, compute="cpu"):
        import coremltools as ct
        from transformers import AutoTokenizer
        self.compute = compute
        self.directory = Path(directory)
        self.config = json.loads((self.directory/"config.json").read_text())
        routes = {"cpu":ct.ComputeUnit.CPU_ONLY,"gpu":ct.ComputeUnit.CPU_AND_GPU,
                  "ane":ct.ComputeUnit.CPU_AND_NE,"all":ct.ComputeUnit.ALL}
        self.models = {}
        for name in ("TextProjector","CodeEmbedder","MultiCodeEmbedder","CodeDecoder","MultiCodeDecoder","SpeechDecoder"):
            route = ct.ComputeUnit.CPU_ONLY if "Embedder" in name or name=="TextProjector" else routes[compute]
            self.models[name] = ct.models.CompiledMLModel(str(self.directory/f"{name}.mlmodelc"), compute_units=route)
        # Preserve Qwen tokenization; the Mistral regex migration does not apply.
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.directory), local_files_only=True, fix_mistral_regex=False)
        self.pad = np.load(self.directory/"tts_pad_embed.npy").reshape(1,-1,1,1)
        self.bos = np.load(self.directory/"tts_bos_embed.npy").reshape(1,-1,1,1)
        self.eos = np.load(self.directory/"tts_eos_embed.npy").reshape(1,-1,1,1)

    def embed(self, name, token):
        return self.models[name].predict({"input_ids":np.array([token], dtype=np.int32)})["input_embeds"].astype(np.float32)

    def prompt(self, text, speaker, language="english"):
        if language not in LANGUAGES:
            raise ValueError(f"Unsupported language: {language}")
        width = self.config["hidden_size"]
        if speaker.size != width or not np.isfinite(speaker).all():
            raise ValueError(f"Expected a finite {width}-dimensional speaker embedding")
        text_embed = lambda t: self.embed("TextProjector", t)
        code_embed = lambda t: self.embed("CodeEmbedder", t)
        prompt = [text_embed(t) for t in self.tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)]
        prompt += [self.pad + code_embed(t) for t in (2154,2156,LANGUAGES[language],2157)]
        prompt += [self.pad + speaker.reshape(1,-1,1,1), self.bos + code_embed(2148)]
        prompt += [text_embed(t) + code_embed(2148) for t in self.tokenizer.encode(text, add_special_tokens=False)]
        prompt += [self.eos + code_embed(2148), self.pad + code_embed(2149)]
        return prompt

    def residual_codes(self, hidden, first, rng, temperature):
        shape=(1,self.config["predictor_kv_dim"],1,16)
        kc=np.zeros(shape,np.float16);vc=np.zeros_like(kc)
        codes=[]; embedding=hidden
        for pos in range(16):
            inputs=step_inputs(embedding,pos,16);inputs.update(key_cache=kc,value_cache=vc)
            output=self.models["MultiCodeDecoder"].predict(inputs)
            kc=output["new_key_cache"];vc=output["new_value_cache"]
            if pos == 0:
                embedding=self.embed("CodeEmbedder",first)
            else:
                token=sample(output["all_logits"][0,pos-1],rng,temperature)
                codes.append(token)
                if pos < 15:
                    embedding=self.embed("MultiCodeEmbedder",(pos-1)*2048+token)
        return codes

    def synthesize(self,text,speaker,language="english",seed=42,max_frames=125,temperature=0.8):
        capacity=self.config["max_seq_len"]
        prompt=self.prompt(text,speaker,language)
        if len(prompt)>=capacity:
            raise ValueError("Text prompt leaves no room in CodeDecoder cache")
        if not 1 <= max_frames <= self.config["speech_decoder_frames"]:
            raise ValueError("max_frames must fit the exported SpeechDecoder; re-export with --speech-frames for longer audio")
        rng=np.random.default_rng(seed)
        cd=self.models["CodeDecoder"];state=cd.make_state() if self.config["stateful"] else None
        if state is None:
            kc=np.zeros((1,self.config["talker_kv_dim"],1,capacity),np.float16);vc=np.zeros_like(kc)
        def forward(embedding,pos):
            nonlocal kc,vc
            inputs=step_inputs(embedding,pos,capacity)
            if state is not None:
                return cd.predict(inputs,state=state)
            inputs.update(key_cache=kc,value_cache=vc)
            output=cd.predict(inputs);kc=output["new_key_cache"];vc=output["new_value_cache"]
            return output
        started=time.perf_counter()
        for pos,embedding in enumerate(prompt):
            output=forward(embedding,pos)
        codes=[];eos=False
        limit=min(max_frames,capacity-len(prompt)+1)
        for frame in range(limit):
            logits=output["logits"].astype(np.float64).reshape(-1)
            for previous in set(row[0] for row in codes):
                logits[previous]=logits[previous]/1.05 if logits[previous]>0 else logits[previous]*1.05
            logits[2048:2150]=-1e9;logits[2151:]=-1e9
            if frame<2:logits[2150]=-1e9
            first=sample(logits,rng,temperature)
            if first==2150:
                eos=True;break
            residual=self.residual_codes(output["hidden_states"],first,rng,temperature)
            codes.append([first]+residual)
            if frame+1<limit:
                embedding=self.embed("CodeEmbedder",first)+self.pad
                for group,token in enumerate(residual):
                    embedding+=self.embed("MultiCodeEmbedder",group*2048+token)
                output=forward(embedding,len(prompt)+frame)
        packed=np.zeros((1,16,self.config["speech_decoder_frames"]),np.int32)
        packed[0,:,:len(codes)]=np.array(codes,dtype=np.int32).T
        audio=self.models["SpeechDecoder"].predict({"audio_codes":packed})["audio"].reshape(-1)
        audio=audio[:len(codes)*self.config["samples_per_frame"]].astype(np.float32)
        if not np.isfinite(audio).all():raise ValueError("Non-finite audio")
        elapsed=time.perf_counter()-started
        metrics={"frames":len(codes),"prompt_positions":len(prompt),"eos":eos,
                 "seconds":len(audio)/24000,"wall_seconds":elapsed,"rtf":elapsed/(len(audio)/24000),
                 "compute":self.compute,"seed":seed}
        return audio,np.array(codes,dtype=np.int32),metrics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory");p.add_argument("text");p.add_argument("--speaker-embedding",required=True)
    p.add_argument("--output",default="speech.wav");p.add_argument("--compute",choices=["cpu","gpu","ane","all"],default="cpu")
    p.add_argument("--max-frames",type=int,default=125);p.add_argument("--seed",type=int,default=42)
    p.add_argument("--language",default="english");p.add_argument("--temperature",type=float,default=0.8)
    a=p.parse_args();pipeline=Pipeline(a.directory,a.compute)
    audio,codes,metrics=pipeline.synthesize(a.text,np.load(a.speaker_embedding),a.language,a.seed,a.max_frames,a.temperature)
    import soundfile as sf
    sf.write(a.output,audio,24000)
    np.save(str(Path(a.output).with_suffix(".codes.npy")),codes)
    Path(a.output).with_suffix(".json").write_text(json.dumps(metrics,indent=2)+"\n")
    print(json.dumps(metrics,indent=2))

if __name__=="__main__":main()
