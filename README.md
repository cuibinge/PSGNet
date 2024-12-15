The code is the core part of a project that does not include training and testing.

Running environment as follows:

	python = 3.9.20
	torch = 2.5.0+cu118
	torchaudio = 2.5.0+cu118
	torchvision = 0.20.0+cu118
Problems:If downloads the model fail due to network issues in Res2Net.py file,you can mannual download it as https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_v1b_26w_4s-3cf99910.pth.
And then,putting it on current file. After, open the line of 195 and 196 while close the line of 197.

	195  # model_state = torch.load('./res2net50_v1b_26w_4s-3cf99910.pth')
	196  # model.load_state_dict(model_state)
	197  lib.load_state_dict(model_zoo.load_url(model_urls['res2net50_v1b_26w_4s']))
