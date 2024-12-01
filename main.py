'''
Run evolution, postprocess results, and plot results
'''
from exo_venus_predict import VenusParameters, predict
from postprocess import postprocess
from plot_full import plot_full
from plot_partial import plot_partial

class GlieseParameters(VenusParameters):
    ''''
    Set parameters for Gliese 12-b
    '''
    def __init__(self):
        super().__init__()
        self.RE = 0.958
        self.ME = 3.87
        self.planet_sep=0.068

        self.num_runs = 720

def run(params: VenusParameters, output_file: str, input_file: str,
                   rerun: bool = True, plot_everything: bool = True):
    '''
    Run the prediction and plot results.
    '''
    if rerun is True:
        predict(params, output_file, input_file)

    inputs, mc_inputs, plotting_outs, post_process_out = postprocess(output_file, input_file)

    if plot_everything:
        # creates 32 plots
        plot_full(inputs, mc_inputs, post_process_out, plotting_outs)
    else:
        # creates 8 plots
        plot_partial(inputs, mc_inputs, post_process_out, plotting_outs)

if __name__ == "__main__":
    PARAMS = GlieseParameters()
    OUT = 'Venus_ouputs_revisions_gliese4'
    IN =  'Venus_ouputs_revisions_gliese4'
    RERUN = False
    run(PARAMS, OUT, IN, rerun=RERUN)
