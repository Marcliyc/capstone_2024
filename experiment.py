from ray.rllib.connectors.env_to_module import FlattenObservations
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.examples.envs.classes.two_step_game import TwoStepGameWithGroupedAgents
from ray.rllib.utils.test_utils import (
    add_rllib_example_script_args,
    run_rllib_example_script_experiment,
)
from environment import MultiAgentStationaryImplicit
from ray.tune.registry import register_env, get_trainable_cls


parser = add_rllib_example_script_args()

if __name__ == "__main__":
    args = parser.parse_args()


    register_env(
        "custom_env",
        lambda config: MultiAgentStationaryImplicit(config),
    )

    base_config = (
        get_trainable_cls(args.algo)
        .get_default_config()
        .environment("custon_env")
        .env_runners(
            env_to_module_connector=lambda env: FlattenObservations(multi_agent=True),
        )
        .multi_agent(
            policies={"p0"},
            policy_mapping_fn=lambda aid, *a, **kw: "p0",
        )
        .rl_module(
            rl_module_spec=MultiRLModuleSpec(
                rl_module_specs={
                    "p0": RLModuleSpec(),
                },
            )
        )
    )

    run_rllib_example_script_experiment(base_config, args)