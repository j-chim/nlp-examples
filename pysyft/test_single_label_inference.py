import syft as sy
import syfertext
from syft.generic.string import String
from syfertext.pipeline.single_label_classifier import AverageDocEncoder, SingleLabelClassifier
from syfertext.local_pipeline import get_test_language_model
import torch
import torch.nn as nn

hook = sy.TorchHook(torch)

me = hook.local_worker
me.is_client_worker = False
alice = sy.VirtualWorker(hook, id = 'alice')
bob = sy.VirtualWorker(hook, id = 'bob')
crypto_provider = sy.VirtualWorker(hook, id = 'crypto_provider')

class Net(sy.Plan):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(300, 10)
        self.fc2 = nn.Linear(10, 2)

    def forward(self, x):
        x = self.fc1(x)
        logits = self.fc2(x)
        return logits

net = Net()
net.build(torch.zeros(size=(1, 300)))

# Working examples:
# Ex1 - run locally without fixed precision 
x = torch.randint(0, 100, (1, 300)).float()
# net(x) 

# Ex 2 - run locally with fixed precision
net_fix_prec = net.fix_precision()
x_fix_prec = x.fix_precision()
net_fix_prec(x_fix_prec)

# Failing examples: both throw syft.exceptions.EmptyCryptoPrimitiveStoreError
# Ex 3 - run encrypted 
net_ptr = net_fix_prec.share(alice, bob, crypto_provider=crypto_provider, protocol="fss")
x_ptr = x_fix_prec.share(alice, bob, crypto_provider=crypto_provider, protocol="fss")
#net_ptr(x_ptr)

# Ex 4 - run SingleLabelClassifier as part of pipeline
nlp = get_test_language_model()

doc_encoder = AverageDocEncoder()
classifier = SingleLabelClassifier(
    classifier=net, 
    doc_encoder=doc_encoder,
    encryption="mpc"
    )  

nlp.add_pipe(classifier, name="classifier", access = {'*'})

string = String('A string to tokenize')

nlp.deploy(worker=bob)

print(nlp.states)

#doc_ptr = nlp(string) 

foo = nlp("Hello!")