#include <torch/torch.h>
#include <iostream>


struct Net : torch::nn::Module {
    torch::nn::Linear fc1{nullptr};
    Net() {
        fc1 = register_module("fc1", torch::nn::Linear(1600, 1));
    }

    template<class T>
    T forward(T x) {
        x = x.reshape(-1);
        x = fc1->forward(x);
        return x;
    }
};


int main() {
    auto net = std::make_shared<Net>();
    torch::optim::SGD optimizer(net->parameters(), 0.01);

    for (size_t epoch = 1; epoch <= 10; epoch++) {
        optimizer.zero_grad();
        auto x = torch::rand({1, 40, 40});
        auto target = torch::randint(0, 1, 1);
        auto prediction = net->forward(x);
        auto loss = torch::l1_loss(prediction, target);
        loss.backward();
        optimizer.step();
        std::cout << "Epoch: " << epoch << " | Loss " << loss.item() << std::endl;
    }
    return 0;
}
