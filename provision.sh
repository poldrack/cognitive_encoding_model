sudo apt-get update
sudo apt-get install gcc
sudo apt-get install python-qt4
sudo apt-get install make
if [ ! -d /home/poldrack/anaconda3 ]
then
   wget https://repo.continuum.io/archive/Anaconda3-4.3.1-Linux-x86_64.sh
   bash Anaconda3-4.3.1-Linux-x86_64.sh
   rm Anaconda3-4.3.1-Linux-x86_64.sh
   export PATH=/home/poldrack/anaconda3/bin:$PATH
   conda update conda
   python -c "import nltk; nltk.download(‘all’)"
fi
if [ ! -d /home/poldrack/cognitive_encoding_model ]
then
  git clone git@github.com:poldrack/cognitive_encoding_model.git
fi

