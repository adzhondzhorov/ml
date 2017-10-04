from app.base import Instance, LearningAlgorithm

from test.acceptance_tests.data import WEATHER_SPORT_PLAY_DATA

def assert_learn(learner: LearningAlgorithm):
    learner.train(WEATHER_SPORT_PLAY_DATA)

    assert learner.predict(Instance(["sunny", "cool", "normal", "FALSE", None]))
    assert not learner.predict(Instance(["rainy", "mild", "high", "TRUE", None]))
