# Installing bazel with Tensorflow and PyTorch.

# 1. Creating a Personal Directory
First, we will a **personal bin** directory where bazel will be located, and will add this path to $PATH.

Create the **personal bin** directory by running the following command:
```
mkdir -p ~/bin
```
Include this **personal bin** directory to searchable path by adding the following lines in **.profile**. 
Often, it has been already added to $PATH, so if this is the case, then you may skip this step:
```
# set PATH so it includes user's private bin if it exists
 if [ -d "$HOME/bin" ] ; then
     PATH="$HOME/bin:$PATH"
 fi
```

# 2. Installing Bazel
Refer to the following page:
https://bazel.build/install

We recommend that you use Bazelisk on Linux Ubuntu.

Refer to the following page regarding the entire installation procedure:
https://github.com/bazelbuild/bazelisk


Download  **bazelisk** in mentioned in the above website.
More specifically, you may download **bazelisk-linux-amd64** from the following page to ~/bin:
https://github.com/bazelbuild/bazelisk/releases


# 3. Making a Bazel Wrapper Script to Use a Local Conda Environment.


One of the philosophical objectives of **Bazel** is hermeticity, which means that the build 
system is independent of the local environment. This property is important for achieving reproducibility.
If the Bazel build process still relies on a local conda environment, it breaks this hermeticity, and reproducibility is not guaranteed in other environments.
However, relying solely on configuration from files like MODULE.bazel can sometimes lead to overly large and heavyweight setups. It might take very long time in initial build since it requires downloading and installing a lot of modules. If we use the **Conda** local virtual environment, even though it breaks hemeticity a little bit, build efficiency becomes better.
So, we will use the following bazel wrapper, which provides **--python_path** and **--test_env**. 


Go to **~/bin**:
```
cd ~/bin
```

Create a text file **bazel** in this directory using a text editor:
```
 #!/bin/bash

 PYTHONBIN=$(python -c "import sys; print(sys.executable)")
 PYTHONPATH=$(python -c "import site; print(site.getsitepackages()[0])")

 bazelisk-linux-amd64 "$@" \
   --python_path="$PYTHONBIN" \
   --test_env=PYTHONPATH="$PYTHONPATH"
```

Changes the mod so that it can be executed:
```
chmod 755 ~/bin/bazel

```

You may find reference about command-line arguments of **Bazel** in the following page:
https://bazel.build/reference/command-line-reference

# 4. Testing whether it is installed correctly.

Before proceeding, first activate your conda environment. For example, if your conda environment is
**py3_11_xai604**, run the following command:
```
conda activate py3_11_xai604
```

If your scrren out is similar to the following screen shot, then **bazel** is installed correctly.
<img src="./run_bazel.png" title="Github_Logo"></img>
