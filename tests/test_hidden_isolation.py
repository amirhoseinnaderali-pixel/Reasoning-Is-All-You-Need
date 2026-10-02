import cttr_vps.frozen_runner as fr

def test_hidden_loader_runs_only_after_condition(monkeypatch):
    order=[]
    def fake_run_condition(*args,**kwargs):
        order.append("condition")
        budget=type("B",(),{"model_calls":1,"debug_steps":0,"token_usage":[],"tokens":lambda s:0,"max_calls":1,"max_tokens":1,"max_debug":0,"max_wall":1,"model_revision":"gemini-2.5-flash","elapsed":lambda s:0})()
        visible={"compiled":True,"tests_passed":1,"tests_failed":0,"total_tests":1,"all_passed":True,"failure_type":"none","tests":[]}
        return "code",["code"],visible,budget,[],0.1,None,1
    class FakeSandbox:
        def evaluate(self,code,tests):
            assert order[0]=="condition"
            order.append("hidden_eval")
            return {"compiled":True,"tests_passed":1,"tests_failed":0,"total_tests":1,"all_passed":True,"failure_type":"none","tests":[]}
    monkeypatch.setattr(fr,"run_condition",fake_run_condition)
    monkeypatch.setattr(fr,"sandbox",lambda protocol:FakeSandbox())
    monkeypatch.setattr(fr,"load_manifest",lambda:{"materialization_sha256":"x","tasks":[]})
    monkeypatch.setattr(fr,"environment_metadata",lambda:{})
    monkeypatch.setattr(fr,"git_sha",lambda:"test")
    monkeypatch.setattr(fr,"config_hash",lambda x:"0"*64)
    monkeypatch.setattr(fr,"dependency_lock_hash",lambda x:"1"*64)
    monkeypatch.setattr(fr,"load_protocol",lambda:{"model":{"model_revision":"gemini-2.5-flash"},"conditions":{"single_pass":{"max_model_calls":1,"max_debug_steps":0}}})
    calls=[]
    def hidden_loader():
        calls.append("hidden_load")
        order.append("hidden_load")
        return [{"input":"","expected_output":""}]
    task={"task_id":"t","visible_tests":[{"input":"","expected_output":""}],"problem":"p"}
    record,_=fr.run_real_or_smoke(task,"single_pass",1,"real","results",hidden_loader=hidden_loader)
    assert calls==["hidden_load"]
    assert order[:2]==["condition","hidden_load"]
    assert record["execution_mode"]=="real"
