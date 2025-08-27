import numpy as np
from function import Function_Calculate as cal, Function_Lib as fun
from function import Single_Band_PA

def ILC(board, ilc_in,logger=None):
    """
    Python implementation of YU_ILC function for DPD technology

    Parameters:
    board: object with transmit method (simulates the board interface)
    ilc_in: dictionary containing:
        - y_d: desired output
        - u_k: initial input
        - nIterations: number of iterations
        - type: 'instantaneous_gain' or 'linear'
        - eta: learning rate (default=0.1)

    Returns:
    dict: containing
        - u_ideal: estimated ideal input
        - PA_Out: PA output when k=1
        - test: error vector
        - NMSE: NMSE vector
        - ILC_final: ILC optimum output
    """
    if logger:
        logger.info(f'================ ILC ==================')
    else:
        print(f'================ ILC ==================')
    # Initialize vectors for Input, Output, error
    In = []  # Will be a list of arrays
    Out = []  # Will be a list of arrays

    u_ideal = ilc_in['u_k'].copy()  # ideal estimated inputs
    u_k = ilc_in['u_k'].copy()

    # Initialize output structure
    ilc_out = {
        'test': np.zeros(ilc_in['nIterations']),
        'NMSE': np.zeros(ilc_in['nIterations']),
        'PA_Out': None,
        'u_ideal': None,
        'ILC_final': None
    }

    for k in range(ilc_in['nIterations']):
        iteration = k + 1  # Python is 0-indexed, MATLAB is 1-indexed
        if logger:
            logger.info(f"Iteration: {iteration}")
        else:
            print(f"Iteration: {iteration}")

        try:
            # Transmit the signal through the board
            y_k = board.transmit(u_k)
            # Normalize the output (different from Fawzy)
            y_k = y_k / np.linalg.norm(y_k) * np.linalg.norm(u_k)
        except Exception as e:
            print(f"Error in transmission: {e}")
            In.append(u_k.copy())
            Out.append(y_k.copy())
            continue

        if k == 0:
            ilc_out['PA_Out'] = y_k  # output for learning

        # Calculate error
        e_k = ilc_in['y_d'] - y_k

        # Update vectors
        ilc_out['test'][k] = np.linalg.norm(e_k)

        # Calculate NMSE, skipping first and last 11 samples
        y_d_trimmed = ilc_in['y_d'][10:-10]  # Python uses 0-indexing, so 10:end-10
        y_k_trimmed = y_k[10:-10]
        ilc_out['NMSE'][k] = cal.nmse(y_d_trimmed, y_k_trimmed,logger,1)
        ACLR = cal.acpr(y_k_trimmed, 100e6, 20e6, 20e6, logger)
        # Store input and output for this iteration
        In.append(u_k.copy())
        Out.append(y_k.copy())

        # Update input based on ILC type
        if ilc_in['type'] == 'instantaneous_gain':
            # Create learning matrix
            learning_matrix = np.diag(y_k / u_k)
            # Update input (using pseudo-inverse for stability)
            u_k = u_k + np.linalg.pinv(learning_matrix) @ e_k
        elif ilc_in['type'] == 'linear':
            u_k = u_k + ilc_in.get('eta', 0.1) * e_k

        # Optional normalization (commented out in original)
        # u_k = u_k / np.linalg.norm(u_k) * np.linalg.norm(ilc_in['y_d'])

        u_ideal = u_k.copy()  # update ideal estimated inputs

    # Find the iteration with minimum NMSE
    k_opt = np.argmin(ilc_out['NMSE'])
    ilc_out['u_ideal'] = In[k_opt]
    ilc_out['ILC_final'] = Out[k_opt]

    return ilc_out, k_opt