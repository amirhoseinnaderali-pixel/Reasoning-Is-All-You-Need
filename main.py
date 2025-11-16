
from planning import planning
import os
import json
from _30step import _30step
from cpp_pipe import phase1_initial_code_generation
from cpp_pipe import optimizer
import asyncio  
from asyncio import run
from time import sleep
def phaze1_planning(problem: str,api_key_google_list: list,api_key_ollama_list: list,list_files: list,output_dir: str):
    planning_results_list=[]
    planning_results_list.append(planning(problem,api_key_google_list[0],api_key_ollama_list[0],list_files[0],previous_planning_results=planning_results_list,output_dir=output_dir))
        
    return planning_results_list

def phaze2_planning_multiple(problem: str,api_key_google_list: list,api_key_ollama_list: list,list_files: list,previous_planning_results: list,output_dir: str):
    planning_results_list=[]
    planning_results_list.append(planning(problem,api_key_google_list[1],api_key_ollama_list[1],list_files[1],previous_planning_results=previous_planning_results,output_dir=output_dir))
        
    return planning_results_list


def phaze3_planning_multiple(problem: str,api_key_google_list: list,api_key_ollama_list: list,list_files: list,previous_planning_results: list,output_dir: str):
    planning_results_list=[]
    planning_results_list.append(planning(problem,api_key_google_list[2],api_key_ollama_list[2],list_files[2],previous_planning_results=previous_planning_results,output_dir=output_dir))
        
    return planning_results_list


def plan_(index : int,output_dir: str):
    with open("ioi_multi_view.json", "r") as f:
        data = json.load(f)
    
    #print(data[0])

    api_key_google_list =[
        "AIzaSyBXC7krDh4mvI4VPKFUHpmkDrEcigOE00o",
        "AIzaSyAiSR_exmQehaC7Q0HPnuQUhr0S9jCtQFs",
        "AIzaSyAVhlHdikARNiTbyJLEBtExGBJPTCWucOg",
        "AIzaSyBt9wnZwb6gGm13gXIDLAs01JuF3PoSnBw",
    ]
    api_key_ollama_list =[
        "66b7ca3198584136a86660733672b5ab.NO-wYz2AeqN7Bf0rRSrLkb0H",
        "3423a52360bf468588b6c80e6957ea1d.nQBGpbmZzCvwWysfb6ORjbGd",
        "77408cf3484946d8bb8cf37220ad2721.837tfoOLJmV3FEka43ozQlZF",
        "9c75046d041a4dca811fd2eaaf3e5696.RH4yyGnyj-qwU8BLCRSr4j7P"


    ]
    list_files=[
        "planning_results1.json",
        "planning_results2.json",
        "planning_results3.json",
        "planning_results4.json",
        "planning_results5.json",
        "planning_results6.json",
        "planning_results7.json",
        
     ]
    
    planning_results_list=[]
    planning_results_list.append(phaze1_planning(data[index]["planning_view"],api_key_google_list,api_key_ollama_list,list_files,output_dir=output_dir))
    print("faze1:planning")
    planning_results_list.append(phaze2_planning_multiple(data[index]["planning_view"],api_key_google_list,api_key_ollama_list,list_files,planning_results_list,output_dir=output_dir))
    print("faze2:planning")
    planning_results_list.append(phaze3_planning_multiple(data[index]["algorithm_view"],api_key_google_list,api_key_ollama_list,list_files,planning_results_list,output_dir=output_dir))
    print("faze3:planning")

        
api_key_google_list =[
        "AIzaSyBXC7krDh4mvI4VPKFUHpmkDrEcigOE00o",
        "AIzaSyAiSR_exmQehaC7Q0HPnuQUhr0S9jCtQFs",
        "AIzaSyAVhlHdikARNiTbyJLEBtExGBJPTCWucOg",
        "AIzaSyBt9wnZwb6gGm13gXIDLAs01JuF3PoSnBw",
    ]
            
import json
#read the ioi_multiple_choice_problems.json file

async def plan_to_code(plan: str, problem: str, output_dir: str,tests: list,speed: int,memory: int):

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    code = []
    phase1_result = await phase1_initial_code_generation(plan, problem,code=code ,num_attempts=20,api_key_google=api_key_google_list[0])
    # Write all codes from list to file, one by one
    with open(os.path.join(output_dir, "phase1.txt"), "w") as f:
        for i, code_item in enumerate(phase1_result.code):
            f.write(f"=== Code Variant {i+1} ===\n")
            f.write(code_item)
            f.write("\n\n")

    # Pass ALL codes from phase1 to phase2 (not just one)
    phase2_result = await phase1_initial_code_generation(plan, problem, phase1_result.code if phase1_result.code else [], num_attempts=20,api_key_google=api_key_google_list[1])
    # Write all codes from list to file, one by one
    with open(os.path.join(output_dir, "phase2.txt"), "w") as f:
        for i, code_item in enumerate(phase2_result.code):
            f.write(f"=== Code Variant {i+1} ===\n")
            f.write(code_item)
            f.write("\n\n")

    # Pass ALL codes from phase2 to phase3 (not just one)
    phase3_result = await phase1_initial_code_generation(plan, problem, phase2_result.code if phase2_result.code else phase1_result.code, num_attempts=20,api_key_google=api_key_google_list[2])
    # Write all codes from list to file, one by one
    with open(os.path.join(output_dir, "phase3.txt"), "w") as f:
        for i, code_item in enumerate(phase3_result.code):
            f.write(f"=== Code Variant {i+1} ===\n")
            f.write(code_item)
            f.write("\n\n")

    # Pass ALL codes from phase3 to phase4 (not just one)
    phase4_result = await phase1_initial_code_generation(plan, problem, phase3_result.code if phase3_result.code else phase2_result.code, num_attempts=20,api_key_google=api_key_google_list[3])
    # Write all codes from list to file, one by one
    with open(os.path.join(output_dir, "phase4.txt"), "w") as f:
        for i, code_item in enumerate(phase4_result.code):
            f.write(f"=== Code Variant {i+1} ===\n")
            f.write(code_item)
            f.write("\n\n")
   
    phase5_result = await phase1_initial_code_generation(plan, problem, phase4_result.code if phase4_result.code else phase3_result.code, num_attempts=20,api_key_google=api_key_google_list[0])
    # Write all codes from list to file, one by one
    with open(os.path.join(output_dir, "phase5.txt"), "w") as f:
        for i, code_item in enumerate(phase5_result.code):
            f.write(f"=== Code Variant {i+1} ===\n")
            f.write(code_item)
            f.write("\n\n")
    
    #code=phase5_result.code
    sleep(100)
    optimizer_result = await optimizerrr(phase5_result.code,output_dir=output_dir)

    # Handle empty optimizer_result
    if not optimizer_result or len(optimizer_result) == 0:
        print("⚠ Warning: optimizer_result is empty, falling back to phase5_result.code")
        if phase5_result.code and len(phase5_result.code) > 0:
            optimizer_result = phase5_result.code
        else:
            print("❌ Error: No code available to proceed with _30step")
            return

    final_result = await _30step(optimizer_result[0], tests,output_dir=output_dir)
    print(final_result)
   

    #all_results =  await _30step(best_code, tests)

async def optimizerrr(code_list: list,output_dir: str): 
    best_code = await optimizer(code_list,output_dir=output_dir)
    return best_code

def  test_pipeline(index: int):
   
  
    with open("ioi_multi_view.json", "r") as f:
        data = json.load(f)


    output_dir=["pipeline_results1","pipeline_results2","pipeline_results3","pipeline_results4","pipeline_results5","pipeline_results6","pipeline_results7",
"pipeline_results8","pipeline_results9","pipeline_results10","pipeline_results11","pipeline_results12","pipeline_results13","pipeline_results14","pipeline_results15","pipeline_results16","pipeline_results17","pipeline_results18","pipeline_results19","pipeline_results20",
"pipeline_results21","pipeline_results22","pipeline_results23","pipeline_results24","pipeline_results25","pipeline_results26","pipeline_results27","pipeline_results28","pipeline_results29","pipeline_results30",
"pipeline_results31","pipeline_results32","pipeline_results33","pipeline_results34","pipeline_results35","pipeline_results36","pipeline_results37","pipeline_results38","pipeline_results39","pipeline_results40",
"pipeline_results41","pipeline_results42","pipeline_results43","pipeline_results44","pipeline_results45","pipeline_results46","pipeline_results47","pipeline_results48","pipeline_results49","pipeline_results50",
"pipeline_results51","pipeline_results52","pipeline_results53","pipeline_results54","pipeline_results55","pipeline_results56","pipeline_results57","pipeline_results58","pipeline_results59","pipeline_results60",
"pipeline_results61","pipeline_results62","pipeline_results63","pipeline_results64","pipeline_results65","pipeline_results66","pipeline_results67","pipeline_results68","pipeline_results69","pipeline_results70",
"pipeline_results71","pipeline_results72","pipeline_results73","pipeline_results74","pipeline_results75","pipeline_results76","pipeline_results77","pipeline_results78","pipeline_results79","pipeline_results80",
"pipeline_results81","pipeline_results82","pipeline_results83","pipeline_results84","pipeline_results85","pipeline_results86","pipeline_results87","pipeline_results88","pipeline_results89","pipeline_results90",
"pipeline_results91","pipeline_results92","pipeline_results93",
    ]

    with open("ioi_multi_view.json", "r") as f:
        data = json.load(f)




        print("start planning")
        output_dir=output_dir[index]
        plan_(index,output_dir=output_dir)
        with open(output_dir+"/planning_results3.json", "r") as f:
            plans = json.load(f)
        print("start code generation")
        sleep(100)
        plan=plans[0]
        problem=data[index]["algorithm_view"]
        code=[]
        tests=data[index]["implementation_view"]["samples"]

        speed=data[index]["implementation_view"]["limits"]["time"]
        memory=data[index]["implementation_view"]["limits"]["memory"]
        asyncio.run(plan_to_code(plan, problem, output_dir,tests,speed,memory))
   



def daily_using():

    tests="""
    ### Input:
 ```
4 5 2
1 2 10 100 5
2 3 10 150 7
3 4 10 120 6
1 3 5 200 8
2 4 8 180 9
1 4 8 500 3
1 4 5 300 2
```

### Expected Output:
```
2
34
3640
1 1->2->4 14
2 1->2->4 14
```
### Input:
```
6 9 4
1 2 20 300 8
2 3 15 250 12
3 6 18 280 10
1 4 10 320 15
4 5 12 290 13
5 6 14 310 11
2 5 8 200 9
3 4 6 180 7
4 6 9 270 14
1 6 25 1000 8
1 6 20 900 7
2 6 15 700 9
3 6 10 600 6
```

### Expected Output:
```
4
1165
29350
1 1->2->3->6 30
2 1->2->5->6 31
3 2->3->6 22
4 3->6 10
```
### Input:
```
8 15 6
1 2 25 400 12
2 3 20 350 15
3 4 18 380 14
4 8 22 420 16
1 5 15 450 18
5 6 17 390 17
6 7 19 410 15
7 8 21 440 13
2 5 12 300 10
3 6 14 320 11
4 7 16 360 12
1 3 10 500 20
5 8 13 480 19
2 7 11 370 14
6 8 15 430 16
1 8 30 1500 10
1 8 25 1400 9
2 8 20 1200 8
3 8 22 1100 7
1 4 18 900 6
5 8 15 800 5
```

### Expected Output:
```
6
2895
89340
1 1->2->3->4->8 57
2 1->2->5->8 49
3 2->3->4->8 45
4 3->4->8 30
5 1->2->3->4 41
6 5->6->7->8 45
```


    """



    problem="""
    # Dynamic Multi-Commodity Flow Network with Time-Cost Constraints

## Problem Statement

You are given a transportation network with **N** cities and **M** bidirectional roads. Each road has the following properties:
- **Capacity**: Maximum number of trucks that can simultaneously use this road
- **Cost**: Cost per truck to traverse this road
- **Time**: Time duration (in minutes) to traverse this road

You have **K** shipments, each needing to travel from its source city to its destination city. Each shipment has:
- Source city (s)
- Destination city (t)
- Volume: Number of trucks required
- Deadline: Maximum time by which it must reach destination
- Priority: Integer between 1 and 10

## Objective

Find an optimal routing plan that:

1. **Maximizes the number of shipments** that reach their destination on time
2. If multiple solutions deliver the same number of shipments, choose the one that **maximizes the sum of (priority × volume)** for delivered shipments
3. If still tied, choose the solution with **minimum total cost**

## Constraints

- 2 ≤ N ≤ 300 (number of cities)
- 1 ≤ M ≤ N×(N-1)/2 (number of roads)
- 1 ≤ K ≤ 100 (number of shipments)
- 1 ≤ capacity ≤ 1000
- 1 ≤ cost ≤ 10^6
- 1 ≤ time ≤ 1000
- 1 ≤ volume ≤ 100
- 1 ≤ deadline ≤ 10000
- 1 ≤ priority ≤ 10
- Time limit: 3 seconds
- Memory limit: 512 MB

## Input Format

**Line 1:** N M K

**Next M lines:** Each line contains `u v cap cost time`
- u, v: Two endpoints of the road (1 ≤ u,v ≤ N)
- cap: Capacity
- cost: Cost per truck
- time: Travel time

**Next K lines:** Each line contains `s t vol deadline priority`
- s: Source city
- t: Destination city
- vol: Volume (number of trucks)
- deadline: Time deadline
- priority: Priority level

## Output Format

**Line 1:** Number of shipments that arrived on time

**Line 2:** Sum of (priority × volume) for delivered shipments

**Line 3:** Total cost

**Following lines:** For each delivered shipment:
#- Route taken (as list of cities)
- Arrival time


    """
    output_dir="Amirhosein_nadeerali"
    api_key_google_list =[
        "AIzaSyBXC7krDh4mvI4VPKFUHpmkDrEcigOE00o",
        "AIzaSyAiSR_exmQehaC7Q0HPnuQUhr0S9jCtQFs",
        "AIzaSyAVhlHdikARNiTbyJLEBtExGBJPTCWucOg",
        "AIzaSyBt9wnZwb6gGm13gXIDLAs01JuF3PoSnBw",
    ]
    api_key_ollama_list =[
        "66b7ca3198584136a86660733672b5ab.NO-wYz2AeqN7Bf0rRSrLkb0H",
        "3423a52360bf468588b6c80e6957ea1d.nQBGpbmZzCvwWysfb6ORjbGd",
        "77408cf3484946d8bb8cf37220ad2721.837tfoOLJmV3FEka43ozQlZF",
        "9c75046d041a4dca811fd2eaaf3e5696.RH4yyGnyj-qwU8BLCRSr4j7P"


    ]
    list_files=[
        "planning_results1.json",
        "planning_results2.json",
        "planning_results3.json",
        "planning_results4.json",
        "planning_results5.json",
        "planning_results6.json",
        "planning_results7.json",
        
     ]
    
    #planning_results_list=[]
    #planning_results_list.append(phaze1_planning(problem,api_key_google_list,api_key_ollama_list,list_files,output_dir=output_dir))
    #print("faze1:planning")
    #planning_results_list.append(phaze2_planning_multiple(problem,api_key_google_list,api_key_ollama_list,list_files,planning_results_list,output_dir=output_dir))
    #print("faze2:planning")
    #planning_results_list.append(phaze3_planning_multiple(problem,api_key_google_list,api_key_ollama_list,list_files,planning_results_list,output_dir=output_dir))
    #print("faze3:planning")
    with open(output_dir+"/planning_results3.json", "r") as f:
        plans=json.load(f)
    plan=plans[0]
    #sleep(100)
    code=[]
    tests=tests.split("\n")
    speed=1
    memory=512
    asyncio.run(plan_to_code(plan, problem, output_dir,tests,speed,memory))
    




if __name__ == "__main__":












    test_pipeline(4)
   
    