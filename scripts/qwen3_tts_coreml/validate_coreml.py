#!/usr/bin/env python3
"""Serial, real-checkpoint conversion validation. Generate fixtures before loading CoreML.

  python validate_coreml.py fixtures --component CodeDecoder --directory bundle --fixtures results
  python validate_coreml.py check --component CodeDecoder --directory bundle --fixtures results --compute cpu
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np

COMPONENTS = ("TextProjector", "CodeEmbedder", "MultiCodeEmbedder", "CodeDecoder", "MultiCodeDecoder", "SpeechDecoder")


def fixtures(args, config):
    import torch
    import convert_coreml as C
    C.apply_patches(); C.patch_rmsnorm()
    torch.set_num_threads(4);torch.set_grad_enabled(False);torch.manual_seed(42)
    name=args.component
    saved={}
    if name != "SpeechDecoder":
        model=C.load_model(config["model_id"],targets=[name],revision=config["revision"])
        talker=model.talker
    if name in COMPONENTS[:3]:
        if name=="TextProjector":w=C.TextProjectorWrapper(talker.model.text_embedding,talker.text_projection);tokens=[151644,77091,198,151671]
        elif name=="CodeEmbedder":w=C.CodeEmbedderWrapper(talker.model.codec_embedding);tokens=[0,2154,2149,2047]
        else:w=C.MultiCodeEmbedderWrapper(talker.code_predictor.model.codec_embedding);tokens=[0,2047,2048,30719]
        for i,token in enumerate(tokens):
            x=torch.tensor([token]);saved[f"{i}_input_ids"]=x.numpy().astype(np.int32)
            saved[f"{i}_input_embeds"]=w(x).numpy()
        saved["count"]=len(tokens)
    elif name in ("CodeDecoder","MultiCodeDecoder"):
        from run_coreml import step_inputs
        cd=name=="CodeDecoder";capacity=config["max_seq_len"] if cd else 16
        positions=sorted(set(p for p in (0,1,2,254,255,256,capacity-2,capacity-1) if 0<=p<capacity)) if cd else list(range(16))
        w=C.CodeDecoderWrapper(talker,stateful=True,max_seq_len=capacity) if cd else C.MultiCodeDecoderWrapper(talker.code_predictor)
        if not cd:
            kc=torch.zeros(1,config["predictor_kv_dim"],1,16);vc=torch.zeros_like(kc)
        for i,pos in enumerate(positions):
            x=torch.randn(1,config["hidden_size"],1,1).half().float()
            inputs=step_inputs(x.numpy(),pos,capacity)
            saved[f"{i}_input_embeds"]=inputs["input_embeds"]
            pad=torch.from_numpy(inputs["key_padding_mask"]).float();update=torch.from_numpy(inputs["kv_cache_update_mask"]).float()
            if cd:out=w(x,torch.tensor([pos]),pad,update)
            else:
                out=w(x,torch.tensor([pos]),kc,pad,update,vc);kc,vc=out[2:]
            saved[f"{i}_logits"]=out[0].numpy();saved[f"{i}_hidden_states"]=out[1].numpy()
        saved["positions"]=positions;saved["count"]=len(positions)
    else:
        from qwen_tts import Qwen3TTSTokenizer
        from huggingface_hub import snapshot_download
        path=snapshot_download(config["tokenizer_id"],revision=config["tokenizer_revision"],allow_patterns=["*.json","*.safetensors"])
        tok=Qwen3TTSTokenizer.from_pretrained(path,device_map="cpu")
        codes=torch.randint(0,2048,(1,16,config["speech_decoder_frames"]))
        # Compare CoreML to the original tokenizer, not only the export wrapper.
        saved["0_audio_codes"]=codes.numpy().astype(np.int32)
        saved["0_audio"]=tok.model.decoder(codes).numpy();saved["count"]=1
    np.savez(Path(args.fixtures)/f"{name}.npz",**saved)
    print(f"Saved {name} real-checkpoint fixtures",flush=True)


def metrics(reference,actual):
    a=np.asarray(reference,dtype=np.float64).reshape(-1);b=np.asarray(actual,dtype=np.float64).reshape(-1)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():raise AssertionError("Invalid model output")
    denominator=np.linalg.norm(a)*np.linalg.norm(b)
    cosine=float(np.dot(a,b)/denominator) if denominator else float(np.array_equal(a,b))
    relative_rmse=float(np.linalg.norm(a-b)/max(np.linalg.norm(a),1e-12))
    return {"cosine":cosine,"relative_rmse":relative_rmse,"max_abs_error":float(np.max(np.abs(a-b)))}


def check(args,config):
    import coremltools as ct
    from run_coreml import step_inputs
    routes={"cpu":ct.ComputeUnit.CPU_ONLY,"gpu":ct.ComputeUnit.CPU_AND_GPU,"ane":ct.ComputeUnit.CPU_AND_NE,"all":ct.ComputeUnit.ALL}
    name=args.component
    model=ct.models.CompiledMLModel(str(Path(args.directory)/f"{name}.mlmodelc"),compute_units=routes[args.compute])
    data=np.load(Path(args.fixtures)/f"{name}.npz")
    cd=name=="CodeDecoder";decoder=cd or name=="MultiCodeDecoder"
    state=model.make_state() if cd and config["stateful"] else None
    capacity=config["max_seq_len"] if cd else 16
    if decoder and state is None:
        dim=config["talker_kv_dim"] if cd else config["predictor_kv_dim"]
        kc=np.zeros((1,dim,1,capacity),np.float16);vc=np.zeros_like(kc)
    results=[];first=None;first_inputs=None
    for i in range(int(data["count"])):
        if decoder:
            pos=int(data["positions"][i]);inputs=step_inputs(data[f"{i}_input_embeds"],pos,capacity)
            if state is None:inputs.update(key_cache=kc,value_cache=vc)
            expected={"logits" if cd else "all_logits":data[f"{i}_logits"],"hidden_states":data[f"{i}_hidden_states"]}
        elif name=="SpeechDecoder":
            inputs={"audio_codes":data[f"{i}_audio_codes"]};expected={"audio":data[f"{i}_audio"]}
        else:
            inputs={"input_ids":data[f"{i}_input_ids"]};expected={"input_embeds":data[f"{i}_input_embeds"]}
        start=time.perf_counter();out=model.predict(inputs,state=state) if state else model.predict(inputs)
        elapsed=time.perf_counter()-start
        if decoder and state is None:kc=out["new_key_cache"];vc=out["new_value_cache"]
        entry={"index":i,"seconds":elapsed,"outputs":{k:metrics(v,out[k]) for k,v in expected.items()}}
        if decoder:entry["position"]=pos
        results.append(entry);print(json.dumps(entry),flush=True)
        if i==0:first={k:v.copy() for k,v in out.items()};first_inputs=inputs.copy()
    reset_error=None
    if state is not None:
        reset=model.predict(first_inputs,state=model.make_state())
        reset_error=float(np.max(np.abs(reset["logits"].astype(np.float32)-first["logits"].astype(np.float32))))
    passed=all(m["cosine"]>=0.999 and m["relative_rmse"]<=0.05 for r in results for m in r["outputs"].values())
    passed=passed and (reset_error is None or reset_error<=1e-3)
    report={"component":name,"compute":args.compute,"passed":passed,"reset_max_abs_error":reset_error,"steps":results}
    path=Path(args.fixtures)/f"{name}-{args.compute}.json";path.write_text(json.dumps(report,indent=2)+"\n")
    if not passed:raise SystemExit(f"Numerical validation failed; see {path}")


def sustain(args, config):
    """Fill every state slot with synthetic inputs, checking finite outputs and reset."""
    import coremltools as ct
    from run_coreml import step_inputs
    if args.compute != "cpu":
        raise ValueError("sustain currently validates CPU execution only")
    if args.component != "CodeDecoder" or not config["stateful"]:
        raise ValueError("sustain requires a stateful CodeDecoder")
    model=ct.models.CompiledMLModel(str(Path(args.directory)/"CodeDecoder.mlmodelc"),
                                   compute_units=ct.ComputeUnit.CPU_ONLY)
    state=model.make_state();rng=np.random.default_rng(42)
    capacity=config["max_seq_len"];started=time.perf_counter();first_logits=None;first_key=None
    for position in range(capacity):
        embed=rng.standard_normal((1,config["hidden_size"],1,1)).astype(np.float16)
        inputs=step_inputs(embed,position,capacity)
        out=model.predict(inputs,state=state)
        if not all(np.isfinite(out[k]).all() for k in ("logits","hidden_states")):
            raise AssertionError(f"Non-finite output at position {position}")
        if position==0:
            first_inputs=inputs;first_logits=out["logits"].copy()
            first_key=state.read_state("key_cache")[...,0].copy()
        if position%128==127:print(f"Completed {position+1}/{capacity} positions",flush=True)
    key=state.read_state("key_cache");value=state.read_state("value_cache")
    assert np.isfinite(key).all() and np.isfinite(value).all()
    assert np.all(np.any(key!=0,axis=(0,1,2)))
    assert np.all(np.any(value!=0,axis=(0,1,2)))
    np.testing.assert_array_equal(key[...,0],first_key)
    reset=model.predict(first_inputs,state=model.make_state())
    np.testing.assert_array_equal(reset["logits"],first_logits)
    report={"component":"CodeDecoder","compute":"cpu","passed":True,
            "consecutive_positions":capacity,"input_kind":"seeded synthetic embeddings",
            "all_outputs_and_states_finite":True,"every_slot_written":True,
            "first_slot_preserved":True,"fresh_state_reset_exact":True,
            "wall_seconds":time.perf_counter()-started}
    (Path(args.fixtures)/"CodeDecoder-sustained-cpu.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("mode",choices=["fixtures","check","sustain"])
    p.add_argument("--component",choices=COMPONENTS,required=True);p.add_argument("--directory",required=True)
    p.add_argument("--fixtures",required=True);p.add_argument("--compute",choices=["cpu","gpu","ane","all"],default="cpu")
    a=p.parse_args();Path(a.fixtures).mkdir(parents=True,exist_ok=True)
    config=json.loads((Path(a.directory)/"config.json").read_text())
    {"fixtures": fixtures, "check": check, "sustain": sustain}[a.mode](a,config)

if __name__=="__main__":main()
