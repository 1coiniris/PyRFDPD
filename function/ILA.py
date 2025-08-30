import numpy as np
from function import Function_Calculate as cal, Function_Lib as fun
from function import Single_Band_PA
import Model.gmp as gmp
import torch
import Model.volterra_nn as MCP_NN
def ILA(board,model, x,nIterations,logger=None):
    """
    Parameters:
    board: object with transmit method (simulates the board interface)

    Returns:
    dict: containing
        - u_ideal: estimated ideal input
        - PA_Out: PA output when k=1
        - test: error vector
        - NMSE: NMSE vector
        - ILC_final: ILC optimum output
    """
    if logger:
        logger.info(f'================ ILA ==================')
    else:
        print(f'================ ILA ==================')

    PA_out = []
    NMSE = []


    y_k = board.transmit(x, logger)
    PA_out.append(y_k)
    nmse_ila = cal.nmse(y_k, x, logger, 1)
    NMSE.append(nmse_ila)

    u_k = x
    for idx in range(nIterations):
        logger.info(f"Start the {idx+1}th iteration")
        if model.name == 'GMP':
            coef = model.model_e(y_k,u_k)
            u_k = model.model_v(x, coef)
            y_k = board.transmit(u_k, logger)
            PA_out.append(y_k)
        if model.name == 'DVR':
            coef = model.DVR_e(y_k, u_k)
            u_k = model.DVR_v(x, coef)
            y_k = board.transmit(u_k, logger)
            PA_out.append(y_k)

        if model.name == 'DVR_NN':
            if idx == 0:
                model.model_train(y_k, u_k, 'tests/20250831/20M/model/OB_DVR_NN_M30_K3_Tanh_202508261803.pt', logger, 1)

            x_coef_tensor = torch.from_numpy(u_k).to(model.device)
            y_coef_tensor = torch.from_numpy(y_k).to(model.device)
            sequences = fun.create_memory_seq(u_k, model.M)  # [N, M+1]
            x_coef_window = torch.from_numpy(sequences).to(model.device)
            y_sequences = fun.create_memory_seq(y_k, model.M)  # [N, M+1]
            y_coef_window = torch.from_numpy(y_sequences).to(model.device)
            coef = model.model_e(y_coef_window, x_coef_tensor)
            u_k = model.apply_dpd(x, coef)
            y_k = board.transmit(u_k, logger)
            PA_out.append(y_k)

        nmse_ila = cal.nmse(y_k,x,logger,1)
        NMSE.append(nmse_ila)
        ACLR = cal.acpr(y_k, board.fs, board.BW, board.BW, logger)
        if NMSE[-1] > NMSE[-2]:
            save = 0
            logger.info('NMSE BAD NonSAVE')
        else:
            save = 1
            model.coef = coef

    ila_out = {
        'NMSE': NMSE,
        'PA_Out': PA_out,
        'model': model
    }


    return ila_out