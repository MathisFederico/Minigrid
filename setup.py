from setuptools import setup, find_packages

setup(
    name='gym_minigrid',
    version='1.0.3',
    keywords='memory, environment, agent, rl, openaigym, openai-gym, gym',
    url='https://github.com/maximecb/gym-minigrid',
    description='Minimalistic gridworld package for OpenAI Gym',
    packages=find_packages(),
    install_requires=[
        'gym>=0.9.6',
        'numpy>=1.15.0'
    ]
)
