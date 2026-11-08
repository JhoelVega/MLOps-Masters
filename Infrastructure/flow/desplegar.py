from infrastructure.flows.entrenamiento_flow import (
    training_flow
)

from infrastructure.flows.inferencia_flow import (
    inference_flow
)


if __name__ == "__main__":

    training_flow.deploy(
        name="training-deployment",
        work_pool_name="default-agent-pool",
    )

    inference_flow.deploy(
        name="daily-inference",
        work_pool_name="default-agent-pool",
        cron="0 8 * * *",
    )
