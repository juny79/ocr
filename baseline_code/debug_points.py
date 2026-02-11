import json
data = json.load(open('outputs/leaderboard_best_for_ensemble/submissions/20260210_225341.json'))
first_img = list(data['images'].values())[0]
first_word = list(first_img['words'].values())[0]
print('Points structure:', first_word['points'][:2])
print('Num points:', len(first_word['points']))
print('First point type:', type(first_word['points'][0]))
