from custom_logging import run

def test_run_custom_logging():
    run(n_training_iterations=1,save=False,plot=False,plot_q_and_gradients=False)
