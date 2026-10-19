"""Contract checks on a small flow model, not pretrained SmolVLA evaluation."""
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest


@unittest.skipUnless(importlib.util.find_spec("torch"), "optional torch")
class InternalTests(unittest.TestCase):
    def setUp(self):
        import torch
        from torch import nn
        from so101_lab.internal_conditioning import InternalConfig, InternalConditioner
        class Flow(nn.Module):
            def __init__(self):
                super().__init__()
                self.model=nn.Module();self.model.action_time_mlp_out=nn.Linear(4,4)
                self.head=nn.Linear(4,2);self.config=SimpleNamespace(compile_model=False)
            def predict_action_chunk(self,batch):
                return self.head(self.model.action_time_mlp_out(batch["noise"]))
            def forward(self,batch):
                return ((self.predict_action_chunk(batch)-batch["action"])**2).mean(),{}
            def reset(self):pass
        torch.manual_seed(18)
        self.policy=Flow();self.adapter=InternalConditioner(InternalConfig(2,4))
        self.batch={"noise":torch.randn(1,3,4),"action":torch.zeros(1,3,2),"observation.state":torch.ones(1,2)}
        self.history=torch.randn(1,4,2);self.mask=torch.ones(1,4,dtype=torch.bool)

    def test_zero_init_preserves_chunk_and_requires_context(self):
        import torch
        from so101_lab.internal_conditioning import InternalBinding
        original=self.policy.predict_action_chunk(self.batch).detach()
        binding=InternalBinding(self.policy,self.adapter)
        with self.assertRaises(RuntimeError):self.policy.predict_action_chunk(self.batch)
        with binding.conditioned(self.history,self.mask):
            torch.testing.assert_close(original,self.policy.predict_action_chunk(self.batch),rtol=0,atol=0)
        binding.close()
        torch.testing.assert_close(original,self.policy.predict_action_chunk(self.batch))

    def test_training_updates_internal_weights_with_frozen_base(self):
        import torch
        from so101_lab.internal_conditioning import InternalBinding,training_loss
        self.policy.requires_grad_(False)
        before={n:p.clone() for n,p in self.policy.named_parameters()}
        binding=InternalBinding(self.policy,self.adapter)
        optimizer=torch.optim.Adam(self.adapter.parameters(),lr=.01)
        initial=self.adapter.residual.weight.clone()
        loss=training_loss(binding,self.batch,self.history,self.mask,
            failure_target=torch.tensor([1]),progress_target=torch.tensor([.7]))
        loss.backward();optimizer.step()
        self.assertFalse(torch.equal(initial,self.adapter.residual.weight))
        for n,p in self.policy.named_parameters():
            if n in before:torch.testing.assert_close(p,before[n],rtol=0,atol=0)
        binding.close()

    def test_causal_padding_reset_and_serialization(self):
        import torch
        from so101_lab.internal_conditioning import InternalChunkPolicy,save,load
        wrapper=InternalChunkPolicy(self.policy,self.adapter)
        wrapper.predict_action_chunk(self.batch);wrapper.predict_action_chunk(self.batch)
        self.assertEqual(len(wrapper.history),2)
        wrapper.reset();self.assertEqual(len(wrapper.history),0)
        mask=self.mask.clone();mask[:,0]=False
        a=self.adapter(self.history,mask)
        history=self.history.clone();history[:,0]=999
        b=self.adapter(history,mask)
        for name in a:torch.testing.assert_close(a[name],b[name],rtol=0,atol=0)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"internal.pt";save(path,self.adapter,{"source":"synthetic_contract_test"})
            restored,metadata=load(path)
            self.assertEqual(metadata["source"],"synthetic_contract_test")
            torch.testing.assert_close(restored(self.history,self.mask)["progress"],self.adapter(self.history,self.mask)["progress"])
            with self.assertRaises(FileExistsError):save(path,self.adapter,{})
        wrapper.binding.close()

    def test_invalid_labels_masks_and_modes_rejected(self):
        import torch
        from so101_lab.internal_conditioning import InternalBinding,training_loss
        from so101_lab.smolvla_live import SmolVLAStream
        mask=self.mask.clone();mask[:,-1]=False
        with self.assertRaises(ValueError):self.adapter(self.history,mask)
        binding=InternalBinding(self.policy,self.adapter)
        with self.assertRaises(ValueError):
            training_loss(binding,self.batch,self.history,self.mask,progress_target=torch.tensor([1.5]))
        binding.close()
        with self.assertRaises(ValueError):SmolVLAStream(self.policy,None,None,adapter=object(),internal_adapter=self.adapter)

    def test_full_trainer_preserves_base_and_episode_split(self):
        import hashlib
        import torch
        from so101_lab.train_internal import Example,train
        history=self.history.clone();history[:,-1]=self.batch["observation.state"]
        rows=[Example(n,self.batch,history,self.mask) for n in ("train","validation")]
        split={"train":["train"],"validation":["validation"]}
        metadata={"source":"synthetic","checkpoint_sha256":hashlib.sha256(b"toy-contract-model").hexdigest(),
            "normalization_fit_episodes":["train"]}
        before={n:p.clone() for n,p in self.policy.named_parameters()}
        with tempfile.TemporaryDirectory() as d:
            result=train(self.policy,self.adapter,rows,split,metadata,Path(d)/"run",epochs=2)
            self.assertEqual(len(result),2)
            self.assertTrue((Path(d)/"run/internal-adapter.pt").is_file())
            with self.assertRaises(ValueError):train(self.policy,self.adapter,rows,
                {"train":["train"],"validation":["train"]},metadata,Path(d)/"invalid")
        for n,p in self.policy.named_parameters():torch.testing.assert_close(p,before[n],rtol=0,atol=0)
        self.assertFalse(hasattr(self.policy,"history_conditioner"))


if __name__=="__main__":unittest.main()
