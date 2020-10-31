import config
import train_test

if __name__ == '__main__':
  configs = config.configurations()
  print ('config:\n', vars(config))
  train_losses, val_losses, bleu_scores = train_test.train_process(configs)