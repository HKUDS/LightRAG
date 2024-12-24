Hi there. Today I'm going to show you how to install the latest version of lollmz and how to configure it and use it.

First, go to the repository, in the release section and download the win_install.bat for windows users, linux_install.sh for linux users and macos_install.sh for macos users.

Make sure you download to a clean folder as the installer will install everything in that folder.

Launch the installer and select your processing unit : A GPU or CPU. Then just wait.

Once the installation is done, just run win_run.bat file. The first time, you will be prompted to select a personal folder. You can put it anywhere you want. Just make sure that the folder has enough space as the models may be very heavy. 

Now before starting, we need to install a binding and a model.
In the UI, go to settings tab, then to bindings zoo section. You can install any one of the bindings, each has its specificity. Here we install GPT4all which uses GGUF file format and support both CPU and GPU.

Next we install a model. Make sure you select a compatible model. For example for GPT4All, it supports Q4 quantization with GGUF format.

Now we select the model and we apply changes then we go to the discussion page.
Now the fun starts.

You can see that the CPU version is slow, so we can go to the configuration of the GPT4All Binding and select GPU as processing unit.


The Generation is now much faster.


We can activate the long term memory option which gives the AI access to the whole discussion database allowing it to remember many things that was said in the past.

Now it is your turn to engage with lollmz and discover its functionalities.

Make sure you subscribe and activate the bell to keep uptodate with my new videos about lollmz. I'm planning on revealing all its secrets in a series of videos so stay tuned.

Thanks for watching
See ya