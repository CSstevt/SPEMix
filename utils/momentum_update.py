def get_params_except_module(model, module_name):
    all_params = model.named_parameters()
    selected_params = [param for name, param in all_params if module_name not in name]
    return selected_params

def update(model, model2, moment):  # to upadte mdoel with the parameters of the mdoel2
    filtered_params = get_params_except_module(model2, 'MP')


    params = model.state_dict()
    params2 = model2.state_dict()

    for name, param1 in params2.items():
        if name in params:
            params2[name]= params2[name] * moment + (1 - moment) *  params[name]