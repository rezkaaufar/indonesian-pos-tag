import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def print_confusion_matrix(confusion_matrix, class_names, figsize=(10, 7), fontsize=14):
  """Prints a confusion matrix, as returned by sklearn.metrics.confusion_matrix, as a heatmap.

  Arguments
  ---------
  confusion_matrix: dictionary
  class_names: list
      An ordered list of class names, in the order they index the given confusion matrix.
  figsize: tuple
      A 2-long tuple, the first value determining the horizontal size of the ouputted figure,
      the second determining the vertical size. Defaults to (10,7).
  fontsize: int
      Font size for axes labels. Defaults to 14.

  Returns
  -------
  matplotlib.figure.Figure
      The resulting confusion matrix figure
  """
  df_cm = pd.DataFrame(
    confusion_matrix#, index=class_names, columns=class_names,
  )
  df_cm = df_cm.fillna(0).astype(int)
  fig = plt.figure(figsize=figsize)
  try:
    heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
  except ValueError:
    raise ValueError("Confusion matrix values must be integers.")
  heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
  heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=90, ha='right', fontsize=fontsize)
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  plt.savefig('confusion_matrix.png')