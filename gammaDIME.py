from keras import backend as K

# gamma-DIME loss
def gamma_dime_loss(args):
    # define the parameter gamma
    gamma = 1
    t_xy = args[0]
    t_xy_bar = args[1]
    loss = -(gamma*K.mean(K.log(t_xy)) - K.mean(K.pow(t_xy_bar, gamma))+1)
    return loss