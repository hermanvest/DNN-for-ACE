class Computeloss:
    def __init__() -> None:
        raise NotImplementedError

    # functions with loss equations corresponding to the ones in the latex document

    # function for checking bounds that if violated, should penalize the neural network
    def penalty_bounds_of_policy(self):
        raise NotImplementedError

    # final loss function that is called form outside of the computeloss function
    def total_loss(self):
        raise NotImplementedError
