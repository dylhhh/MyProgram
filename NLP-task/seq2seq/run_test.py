import config
import train_test

# 在執行 Test 之前，請先行至 config 設定所要載入的模型位置

if __name__ == '__main__':
  configs = config.configurations()
  print ('config:\n', vars(configs))
  test_loss, bleu_score = train_test.test_process(configs)
  print (f'test loss: {test_loss}, bleu_score: {bleu_score}')