import torch
from torch import nn, optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import zipfile
import os

class Classifier:
  def __init__(self):
    torch.manual_seed(123)
    #Constroi o modelo da rede neural
    #output = (input - filter + 1) / stride
    self.classifier = nn.Sequential(nn.Conv2d(in_channels = 3, 
                                        out_channels = 32,
                                        kernel_size = 3),
                              nn.ReLU(),
                              nn.BatchNorm2d(num_features = 32),
                              #(64 - 3 + 1) / 1 = 62x62
                              nn.MaxPool2d(kernel_size = 2),
                              #31x31
                              nn.Conv2d(32, 32, 3),
                              nn.ReLU(),
                              nn.BatchNorm2d(32),
                              #(31 - 3 + 1) / 1 = 29x29
                              nn.MaxPool2d(2),
                              #14x14
                              nn.Flatten(),
                              #6272 -> 128 -> 128 -> 1
                              nn.Linear(in_features = 14*14*32, out_features = 128),
                              nn.ReLU(),
                              nn.Dropout(0.2),
                              nn.Linear(128, 128),
                              nn.ReLU(),
                              nn.Dropout(0.2),
                              nn.Linear(128, 1),
                              nn.Sigmoid())

    self.criterion = nn.BCELoss()
    self.optmizer = optim.Adam(self.classifier.parameters())

    #Base de dados de treinamento
    self.data_dir_training = os.path.expanduser('~') + '/Plasmodium-Classifier/plasmodium_images/dataset/train_set'

    #Transforma as imagens para que após o treinamento a rede neural também consiga 
    #classificar imagens com qualidade ruim, ou distorcidas.
    self.transform_train = transforms.Compose(
        [
        #Dimensiona todas as imagem para ter o mesmo tamanho. 
        transforms.Resize([64, 64]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees = 7, translate = (0, 0.07), shear = 0.2, scale = (1, 1.2)),
        #transforma para o formato tensor
        transforms.ToTensor()
        ]
    )

    self.transform_test = transforms.Compose(
        [
        #Dimensiona todas as imagem para ter o mesmo tamanho.
        transforms.Resize([64, 64]),
        #transforma para o formto tensor
        transforms.ToTensor()
        ]
    )

    #Cria o dataset para treinamento
    self.training_dataset = datasets.ImageFolder(self.data_dir_training, transform = self.transform_train)
    self.training_loader = torch.utils.data.DataLoader(self.training_dataset, batch_size = 32, shuffle = True)

    self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    self.classifier.to(self.device)
    self.training()

  #Função de treinamento da rede
  def training_loop(self,loader, epoch):
      running_loss = 0.
      running_accuracy = 0.
      
      for i, data in enumerate(loader):
          inputs, labels = data
          inputs, labels = inputs.to(self.device), labels.to(self.device)
          
          self.optmizer.zero_grad()        
          outputs = self.classifier(inputs)
          
          loss = self.criterion(outputs, labels.float().view(*outputs.shape))
          loss.backward()
          
          self.optmizer.step()

          running_loss += loss.item()

          predicted = torch.tensor([1 if output > 0.5 else 0 for output in outputs]).to(self.device)
          
          equals = predicted == labels.view(*predicted.shape)
          
          accuracy = torch.mean(equals.float())
          running_accuracy += accuracy
                    
          #Imprimindo os dados referentes a esse loop
          print('\r ÉPOCA {:3d} - Loop {:3d} de {:3d}: perda {:03.2f} - precisão {:03.2f}'.format(epoch + 1, i + 1, len(loader), loss, accuracy), end = '\r')
          
      #Imprimindo os dados referentes a essa época
      print('\r ÉPOCA {:3d} FINALIZADA: perda {:.5f} - precisão {:.5f}'.format(epoch + 1, running_loss/len(loader), running_accuracy/len(loader)))

  # Quanto maior a quantidade de epocas de treinamento, maior a precisão da rede na classificação
  # Quantitade minima em produção = 10
  # Quantidade para agilizar desenvolvimento = 2 
  def training(self):
     for epoch in range(10):
      print('Treinando...')
      self.training_loop(self.training_loader, epoch)
      self.classifier.eval()
      self.classifier.train()


  #Função de classificação da imagem inserida
  def classify_image(self, fname):
    from PIL import Image
    image = Image.open(fname)
    
    image = image.resize((64, 64))
    image = np.array(image.getdata()).reshape(*image.size, 3)
    image = image / 255
    image = image.transpose(2, 0, 1)
    image = torch.tensor(image, dtype=torch.float).view(-1, *image.shape)

    self.classifier.eval()
    image = image.to(self.device)
    output = self.classifier.forward(image)
    #Se a previsão de Uninfected for maior que 50%
    if output > 0.5:
      #Calcula a probabilidade para exibir
      output = str(output)
      output = output.split('[[')[1]
      output = output.split(']]')[0]
      probability = float(output) * 100
      probability = round(probability, 2)

      result = {
        'class': 0,
        'probability':probability,
        'classification': 'NÃO PARASITADO'
      } 
    #Caso a previsão de Uninfected for menor que 50% a rede neural classifica como Parasitized 
    else:
      #Calcula a probabilidade para exibir
      output = str(output)
      output = output.split('[[')[1]
      output = output.split(']]')[0]
      probability = float(output) * 100
      probability = 100 - probability
      probability = round(probability, 2)
      
      result = {
        'class': 1,
        'probability':probability,
        'classification': 'PARASITADO'
      }
      
    return result
    