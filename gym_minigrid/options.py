
class Option():

    def __call__(self, observations, greedy: bool=False):
        """ Use the option to get next actions.

        Args:
            observations: Observations of the environment.
            greedy: If true, the agent should act greedily.

        Returns:
            actions: Actions given by the option with current observations.
            option_done: True if the option is done, False otherwise.

        """
        raise NotImplementedError


