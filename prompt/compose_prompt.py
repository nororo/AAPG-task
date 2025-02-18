

# qag
def make_prompt_qag_prep(prompt_dict:dict,sample_text:str):
    instruction = prompt_dict["qag_instruction"]
    
    constraints = prompt_dict["qag_constraints"]
    preface_constraint = "#### 注意事項"
    bullet_char = " * "
    constraint = preface_constraint+"\n"+bullet_char+("\n"+bullet_char).join(constraints)
    
    output_format = prompt_dict["qag_output_formats"]
    system_prompt_comp = instruction+"\n\n"+constraint
    user_prompt_comp = "#### 文章"+"\n"+sample_text+"\n\n"+output_format+"\n\n"+"#### 回答" +"\n"
    return system_prompt_comp, user_prompt_comp

def make_prompt_qag(prompt_dict:dict,sample_text:str):
    instruction = prompt_dict["qag_instruction"]
    
    constraints = prompt_dict["qag_constraints"]
    preface_constraint = "#### 注意事項"
    bullet_char = " * "
    constraint = preface_constraint+"\n"+bullet_char+("\n"+bullet_char).join(constraints)
    
    output_format = prompt_dict["qag_output_formats"]
    system_prompt_comp = instruction+"\n\n"+constraint+"\n\n"+output_format
    user_prompt_comp = "#### 文章"+"\n"+sample_text+"\n\n"+"#### 評価結果"
    return system_prompt_comp, user_prompt_comp

def make_prompt_eval_add_comp(prompt_dict:dict,ans_text,pred_text):
    instruction = prompt_dict["add_comp_instruction"]
    #scale_dict = prompt_dict["e_comp_scale"]

    #preface_scale = "次の評価スケールによって得点を決定します。\n\n#### 評価スケール"
    #scale = "\n".join([f"{key}:{val}" for key,val in scale_dict.items()])

    constraints = prompt_dict["add_comp_constraints"]
    preface_constraint = "#### 注意事項"
    bullet_char = " * "
    constraint = preface_constraint+"\n"+bullet_char+("\n"+bullet_char).join(constraints)

    output_format = prompt_dict["add_comp_output_format"]
    example_text = prompt_dict["add_comp_example_text"]

    system_prompt_comp = instruction+"\n\n"+constraint+"\n\n"+output_format+"\n\n"+example_text
    user_prompt_comp = f"#### ベース文章\n{pred_text}\n\n#### 追加候補の文章\n{ans_text}"+"\n\n"+"#### 回答" +"\n"
    return system_prompt_comp, user_prompt_comp


def make_prompt_eval_add_comp_ess(prompt_dict:dict,ans_text,pred_text):
    instruction = prompt_dict["add_comp_ess_instruction"]
    #scale_dict = prompt_dict["e_comp_scale"]

    #preface_scale = "次の評価スケールによって得点を決定します。\n\n#### 評価スケール"
    #scale = "\n".join([f"{key}:{val}" for key,val in scale_dict.items()])

    constraints = prompt_dict["add_comp_ess_constraints"]
    preface_constraint = "#### 注意事項"
    bullet_char = " * "
    constraint = preface_constraint+"\n"+bullet_char+("\n"+bullet_char).join(constraints)

    output_format = prompt_dict["add_comp_ess_output_format"]
    example_text = prompt_dict["add_comp_ess_example_text"]

    system_prompt_comp = instruction+"\n\n"+constraint+"\n\n"+output_format+"\n\n"+example_text
    user_prompt_comp = f"#### ベース文章\n{pred_text}\n\n#### 追加候補の文章\n{ans_text}"+"\n\n"+"#### 回答" +"\n"
    return system_prompt_comp, user_prompt_comp



def make_prompt_eval_comp(prompt_dict:dict,ans_text,pred_text):
    instruction = prompt_dict["e_comp_instruction"]
    scale_dict = prompt_dict["e_comp_scale"]

    preface_scale = "次の評価スケールによって得点を決定します。\n\n#### 評価スケール"
    scale = "\n".join([f"{key}:{val}" for key,val in scale_dict.items()])

    constraints = prompt_dict["e_comp_constraints"]
    preface_constraint = "#### 注意事項"
    bullet_char = " * "
    constraint = preface_constraint+"\n"+bullet_char+("\n"+bullet_char).join(constraints)

    output_format = prompt_dict["e_comp_output_formats"]
    example_text = prompt_dict["e_comp_example_text"]

    system_prompt_comp = instruction+"\n\n"+preface_scale+"\n"+scale+"\n\n"+constraint+"\n\n"+output_format+"\n\n"+example_text
    user_prompt_comp = f"#### 正解\n{ans_text}\n\n#### 予測された回答\n{pred_text}"+"\n\n"+"#### 評価結果"
    return system_prompt_comp, user_prompt_comp


def make_prompt_eval_comp2(prompt_dict:dict,ans_text,pred_text):
    instruction = prompt_dict["e_comp2_instruction"]
    scale_dict = prompt_dict["e_comp2_scale"]

    preface_scale = "次の評価スケールによって得点を決定します。\n\n#### 評価スケール"
    scale = "\n".join([f"{key}:{val}" for key,val in scale_dict.items()])

    constraints = prompt_dict["e_comp2_constraints"]
    preface_constraint = "#### 注意事項"
    bullet_char = " * "
    constraint = preface_constraint+"\n"+bullet_char+("\n"+bullet_char).join(constraints)

    output_format = prompt_dict["e_comp2_output_formats"]
    example_text = prompt_dict["e_comp2_example_text"]

    system_prompt_comp = instruction+"\n\n"+preface_scale+"\n"+scale+"\n\n"+constraint+"\n\n"+output_format+"\n\n"+example_text
    user_prompt_comp = f"#### 正解\n{ans_text}\n\n#### 予測された回答\n{pred_text}"+"\n\n"+"#### 評価結果"
    return system_prompt_comp, user_prompt_comp

def make_prompt_eval_conc(prompt_dict:dict,description,ans_text,pred_text):
    instruction = prompt_dict["e_conc_instruction"]
    scale_dict = prompt_dict["e_conc_scale"]

    preface_scale = "次の評価スケールによって得点を決定します。\n#### 評価スケール"
    scale = "\n".join([f"{key}:{val}" for key,val in scale_dict.items()])

    constraints = prompt_dict["e_conc_constraints"]
    preface_constraint = "#### 注意事項"
    bullet_char = " * "
    constraint = preface_constraint+"\n"+bullet_char+("\n"+bullet_char).join(constraints)
    output_format = prompt_dict["e_conc_output_formats"]
    example_text = prompt_dict["e_conc_example_text"]

    #eval_text="""{"得点": 4}"""
    #context=f"#### 検討事項\n{description}\n\n#### 例\n##### 回答:\n{ans_text}\n\n##### 採点結果\n{eval_text}"
    end_instruction="以上の指示に基づいて、提供された予測された回答を評価し、適切な得点を付けてください。"
    
    system_prompt_comp = instruction+"\n\n"+preface_scale+"\n"+scale+"\n\n"+constraint+"\n\n"+output_format+"\n\n"+example_text+"\n\n"+end_instruction
    
    user_prompt_comp = f"#### 検討事項\n{description}\n\n#### 予測された回答:\n{pred_text}"+"\n\n"+"#### 評価結果"
    return system_prompt_comp, user_prompt_comp

def make_prompt_eval_rel(prompt_dict:dict,description,ans_text,pred_text):
    instruction = prompt_dict["e_rel_instruction"]
    scale_dict = prompt_dict["e_rel_scale"]

    preface_scale = "次の評価スケールによって得点を決定します。\n#### 評価スケール"
    scale = "\n".join([f"{key}:{val}" for key,val in scale_dict.items()])

    constraints = prompt_dict["e_rel_constraints"]
    preface_constraint = "#### 注意事項"
    bullet_char = " * "
    constraint = preface_constraint+"\n"+bullet_char+("\n"+bullet_char).join(constraints)
    output_format = prompt_dict["e_rel_output_formats"]
    example_text = prompt_dict["e_rel_example_text"]

    #system_prompt_comp = instruction+"\n"+preface_scale+"\n"+scale+"\n"+constraint+"\n\n"+output_format

    #eval_text="""{"得点": 4}"""
    #context=f"#### 検討事項\n{description}\n\n#### 例\n##### 回答:\n{ans_text}\n\n##### 採点結果\n{eval_text}"
    end_instruction="以上の指示に基づいて、提供された予測された回答を評価し、適切な得点を付けてください。"
    
    system_prompt_comp = instruction+"\n\n"+preface_scale+"\n"+scale+"\n\n"+constraint+"\n\n"+output_format+"\n\n"+example_text+"\n\n"+end_instruction
    
    user_prompt_comp = f"#### 検討事項\n{description}\n\n#### 予測された回答:\n{pred_text}"+"\n\n"+"#### 評価結果"

    #user_prompt_comp = f"#### 検討事項\n{description}\n#### 例\n##### 回答:\n{ans_text}\n##### 採点結果\n{eval_text}\n\n#### 予測された回答:\n{pred_text}"
    return system_prompt_comp, user_prompt_comp

def make_prompt_eval_fluent(prompt_dict:dict,description,ans_text,pred_text):
    instruction = prompt_dict["e_fluent_instruction"]
    scale_dict = prompt_dict["e_fluent_scale"]

    preface_scale = "次の評価スケールによって得点を決定します。\n#### 評価スケール"
    scale = "\n".join([f"{key}:{val}" for key,val in scale_dict.items()])

    constraints = prompt_dict["e_fluent_constraints"]
    preface_constraint = "#### 注意事項"
    bullet_char = " * "
    constraint = preface_constraint+"\n"+("\n"+bullet_char).join(constraints)

    system_prompt_comp = instruction+"\n"+preface_scale+"\n"+scale+"\n"+constraint
    user_prompt_comp = f"#### 検討事項\n{description} ####\n例\n##### 回答:\n{ans_text}\n##### 採点結果: 5\n\n#### 予測された回答:\n{pred_text}"+"\n\n"+"#### 評価結果"
    return system_prompt_comp, user_prompt_comp

def make_prompt_eval_hal(prompt_dict:dict,description,ans_text,pred_text):
    instruction = prompt_dict["e_hal_instruction"]
    scale_dict = prompt_dict["e_hal_scale"]

    preface_scale = "次の評価スケールによって得点を決定します。\n#### 評価スケール"
    scale = "\n".join([f"{key}:{val}" for key,val in scale_dict.items()])

    constraints = prompt_dict["e_hal_constraints"]
    preface_constraint = "#### 注意事項"
    bullet_char = " * "
    constraint = preface_constraint+"\n"+bullet_char+("\n"+bullet_char).join(constraints)
    output_format = prompt_dict["e_hal_output_formats"]
    example_text = prompt_dict["e_hal_example_text"]

    #system_prompt_comp = instruction+"\n"+preface_scale+"\n"+scale+"\n"+constraint+"\n\n"+output_format

    #eval_text="""{"得点": 4}"""
    #context=f"#### 検討事項\n{description}\n\n#### 例\n##### 回答:\n{ans_text}\n\n##### 採点結果\n{eval_text}"
    end_instruction="以上の指示に基づいて、提供された予測された回答を評価し、適切な得点を付けてください。"
    
    system_prompt_comp = instruction+"\n\n"+preface_scale+"\n"+scale+"\n\n"+constraint+"\n\n"+output_format+"\n\n"+example_text+"\n\n"+end_instruction
    
    user_prompt_comp = f"#### 検討事項\n{description}\n\n#### 予測された回答:\n{pred_text}"+"\n\n"+"#### 評価結果"

    #user_prompt_comp = f"#### 検討事項\n{description}\n#### 例\n##### 回答:\n{ans_text}\n##### 採点結果\n{eval_text}\n\n#### 予測された回答:\n{pred_text}"
    return system_prompt_comp, user_prompt_comp