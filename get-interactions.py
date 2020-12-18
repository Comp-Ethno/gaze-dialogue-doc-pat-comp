import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

videos= ["sample"]

for x in videos:

	speech = pd.read_csv(('classifications/speech-'+x+'.csv'))
	gaze = pd.read_csv(('classifications/gaze-'+x+'.csv'))

	clock = speech['clock'].values

	interactions = np.add((speech['speech'].values)*2, gaze['gaze'].values)
		# 0 -> ?, 1 -> DC, 2 -> DP, 3 -> DPC

	interactions_dict = {'clock': clock,
					'interaction': interactions
					}
	interactions_numbers_df = pd.DataFrame (interactions_dict, columns = ['clock','interaction'])
	pd.DataFrame(interactions_numbers_df).to_csv((('classifications/interactions-'+x+'.csv')), index=False)


	def plot_speech_gaze():
		ax = plt.gca()
		ax.set(ylim=(-0.1, 1.1))
		speech.plot(drawstyle="steps-post", x='clock', y='speech', ax=ax)
		gaze.plot(drawstyle="steps-post", x='clock', y='gaze', ax=ax)
		plt.show()

	def plot_interaction():
		ax = plt.gca()
		ax.set(ylim=(-0.1, 3.1))

		plt.yticks(np.arange(0, 3, step=1.0))  # Set label locations.
		plt.yticks(np.arange(4), ['Unknown', 'Doctor-Computer', 'Doctor-Patient', 'Doctor-Patient-Computer'])  # Set text labels.

		interactions_numbers_df.plot(drawstyle="steps-post", x='clock', y='interaction', ax=ax)
		plt.show()

	#plot_speech_gaze()
	plot_interaction()