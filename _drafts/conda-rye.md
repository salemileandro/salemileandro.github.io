---
title: Managing Python Environments and Dependencies
tags: [Python, Software Development]
style: 
color: primary
description: Let's discuss the best tools for managing Python environments and dependencies with ease!
---
<div class="toc-container">
    <h2>Table of Contents</h2>
    <ul>
        <li><a href="#1-introduction">1. Introduction</a></li>
        <li>
            <a href="#2-miniforge-conda">2. Miniforge conda</a>
            <ul>
                <li><a href="#21-installing-miniforge-conda">2.1 Installing Miniforge conda</a></li>
                <li><a href="#22-creating-and-managing-environments-with-miniforge-conda">2.2 Creating and Managing Environments with Miniforge conda</a></li>
                <li><a href="#23-installing-dependencies-with-miniforge-conda">2.3 Installing Dependencies with Miniforge conda</a></li>
            </ul>
        </li>
        <li>
            <a href="#3-rye">3. Rye</a>
            <ul>
                <li><a href="#31-installing-rye">3.1 Installing Rye</a></li>
                <li><a href="#32-creating-an-environment-with-rye">3.2 Creating an Environment with Rye</a></li>
                <li><a href="#33-installing-dependencies-with-rye">3.3 Installing Dependencies with Rye</a></li>
            </ul>
        </li>
        <li><a href="#4-miniforge-conda-or-rye">4. Miniforge conda or Rye?</a></li>
    </ul>
</div>

# 1. Introduction

Managing Python environments is key to avoiding compatibility headaches and keeping workflows smooth. Each project often needs specific Python versions and libraries, and juggling them manually can quickly get messy. Environment management tools solve this by isolating dependencies, ensuring every project has exactly what it needs.


With so many tools out there—*venv, virtualenv, Conda, pyenv, Poetry, ...*—it's easy to feel overwhelmed. In this post, I'll introduce two tools that, in my view, strike a great balance between simplicity and functionality: Miniforge (conda) and Rye. I will assume that you're on a UNIX-based computer (Linux or MacOS), but Windows users can follow this post using WSL.


# 2. Miniforge conda
[Miniforge](https://github.com/conda-forge/miniforge) is a lightweight, open-source implementation of conda, designed specifically to work with the [conda-forge](https://conda-forge.org/) ecosystem. Unlike the full Anaconda distribution, which bundles numerous packages and tools, Miniforge focuses on providing a minimal, streamlined environment. The base environment's packages come exclusively from conda-forge, an extensive, open-source, community-maintained repository of conda packages, and the conda-forge channel is set as the default. In simple terms, Miniforge is the open-source, community-driven alternative to Anaconda.
 

<div class="callout callout-tip">
  <div>
  Anaconda has long been a popular choice for Python users, but recent changes to its licensing model have sparked concerns in the community. As of March 2024, organizations with 200 or more employees are required to purchase a license to use Anaconda's software, including access to its default package channels. This policy shift has impacted not just large companies but also academic and research institutions that previously relied on Anaconda under the assumption that it was freely available. For example, the University of Virginia recently transitioned away from Anaconda due to these changes (read more <a href="https://web.archive.org/web/20241202191039/https://www.rc.virginia.edu/2024/10/transition-from-anaconda-to-miniforge-october-15-2024/">here</a>). Given these changes, I recommend Miniforge as a more open-source-friendly, lightweight, and license-free alternative.
  </div>
</div>

## 2.1 Installing Miniforge conda

Installing Miniforge is straightforward. As explained in their [github repo](https://github.com/conda-forge/miniforge), you can 
download and execute the installation script using:

```bash
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
```

During the installation, you'll be prompted to follow a few steps. Toward the end, Miniforge will ask if you *"wish to update your shell profile to automatically initialize conda?"* I recommend saying yes, as this step ensures that conda is ready to use whenever you open a terminal. It will add the following block to your .bashrc (or .zshrc if you're using the zsh shell like me 🤓):

```bash
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/username/miniforge3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/username/miniforge3/etc/profile.d/conda.sh" ]; then
        . "/home/username/miniforge3/etc/profile.d/conda.sh"
    else
        export PATH="/home/username/miniforge3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
```
You shouldn't manually modify this block—it's managed entirely by Miniforge. After restarting your shell, two things happen:

1. You'll now have access to the conda command.
2. The base environment will activate automatically every time you start a new shell.

To disable this auto-activation (trust me, you'll thank me later), run:

```bash
conda config --set auto_activate_base false
```
Restart your shell once more, and you're all set! You can verify the installation by running `conda info`, which should produce output like this:

```bash
     active environment : None
            shell level : 0
       user config file : /home/username/.condarc
 populated config files : /home/username/miniforge3/.condarc
          conda version : 24.9.2
    conda-build version : not installed
         python version : 3.12.7.final.0
                 solver : libmamba (default)
```


## 2.2 Creating and Managing Environments with Miniforge conda
Now that you've got Miniforge conda set up, you can start creating isolated environments 
for your projects. These environments act as self-contained spaces that store everything 
your project needs, e.g., Python version and specific library versions.

To create a new environment, use the conda create command. For example, if you want to 
create an environment called `myenv` with Python 3.12, run:

```bash
conda create -n myenv python=3.12 pip
```
Here's what this command does:

* `-n myenv` specifies the environment name (myenv).
* `python=3.12` ensures the environment uses Python 3.12.
* `pip` adds pip explicitly to the environment. While conda usually installs pip alongside Python, it’s good practice to specify it explicitly to avoid surprises.


To activate your new environment, use:

```bash
conda activate myenv
```

Once activated, the shell prompt will show the environment's name, indicating you're working within that environment. You can also use ``conda env list`` to list all the available environments, along with their paths.

<div class="callout callout-info">
  <div>
    When you activate an environment, several things happen:
    <ul>
      <li>
        The environment's bin directory is moved to the front of your PATH, ensuring commands like 
        <code>python</code> and <code>pip</code> use the environment-specific versions.
      </li>
      <li>
        Environment variables like <code>CONDA_PREFIX</code> are set to help tools locate 
        dependencies.
      </li>
      <li>
        Your shell prompt updates (e.g., from <code> $></code> to  <code>(myenv)$></code>), so you know which 
        environment is active.
      </li>
      <li>
        Everything you do—installing packages, running scripts—stays isolated to that 
        environment.
      </li>
    </ul>
  </div>
</div>

When you activate an environment, its bin directory is moved to the front of your PATH.
This means commands like `pip install numpy` will install numpy directly into your 
`myenv` environment. Take care that this only works as intented if you've installed pip 
within the environment. Normally, pip is installed alongside Python, so this problem
may occur if you've created a bare environment without Python/pip—for 
example, using `conda create -n myenv`—then running `pip install ...`. This might use the 
globally installed pip, which could lead to confusion or unexpected behavior.
Not sure which pip you're using? A quick check with `which pip` will tell you exactly 
which one is being resolved! 


To deactivate the environment, use:

```bash
conda deactivate
```

And if an environment is no longer needed, remove it with:

```bash
conda remove -n myenv --all
```

The ``--all`` flag ensures everything associated with the environment is deleted.

## 2.3 Installing Dependencies with Miniforge Conda
With your environment ready, it’s time to add the libraries your project needs. Start by activating your environment using `conda activate myenv`, then, use the conda install command to install dependencies. For example:

```bash
conda install 'numpy=2.0' 'pandas>=2.0' matplotlib
```
<div class="callout callout-warning">
  <div>
    Be mindful of the difference between conda and pip version pinning. While conda uses one equal sign (numpy=2.0), pip requires two (numpy==2.0)! 
  </div>
</div>

Sometimes, you’ll need libraries that aren’t available through conda-forge. In such cases, you can use pip, but **always install pip packages after your conda dependencies**. This ensures conda resolves its dependencies first, reducing conflicts. To install pip packages within your active environment:

```bash
pip install fastapi uvicorn[standard]
```



#### Ensuring Reproducibility with an `environment.yaml` File

Once you’ve installed all the required libraries using conda install and pip install, it’s time to save the environment configuration to ensure reproducibility. You can do this with the following command:

```bash
conda env export --no-builds > environment.yaml
```

This command generates a detailed file that includes everything installed in your environment—Python version, libraries, and their dependencies. It will also have a pip section with the libraries installed via pip (if you have used it). The `--no-builds` option removes platform-specific build details. This ensures that the environment can be recreated on different operating systems. If everyone on your team uses the same platform (e.g., Linux), you might omit this flag, but I strongly recommend keeping it for portability.

Here’s a truncated example of what the exported `environment.yaml` file might look like:

```yaml
name: myenv
channels:
  - conda-forge
dependencies:
  - _libgcc_mutex=0.1=conda_forge
  - _openmp_mutex=4.5=2_gnu
  - alsa-lib=1.2.13=hb9d3cd8_0
  - brotli=1.1.0=hb9d3cd8_2
  ...
  - pip:
      - annotated-types==0.7.0
      - anyio==4.8.0
      ...
```
Commit this file to version control (e.g., Git) so your collaborators or future self can recreate the exact environment with:

```bash
conda env create -f environment.yaml
```

Once exported, add the `environment.yaml` file to version control (e.g., Git) so it can be shared or reused! The environment can then be recreated using:

```bash
conda env create -f environment.yaml
```


If you need to add or remove dependencies later, you can:

* Add libraries using `conda install` or `pip install`.
* Remove libraries using `conda remove` or `pip uninstall`.

After making changes, don't forget to re-export the environment with:

```bash
conda env export --no-builds > environment.yaml
```

And commit it to version control!



# 3. Rye

[Rye](https://rye.astral.sh/) is a modern, lightweight tool for managing Python environments, dependencies, and even project initialization. What sets Rye apart is its strong reliance on the pyproject.toml file, which has become the standard for defining Python projects. This means you won’t need to deal with additional environment files—everything is streamlined into one place.


Rather than reinventing the wheel, Rye acts as a wrapper around existing tools like pip (or [uv](https://docs.astral.sh/uv/)), using them effectively to get the job done. I’ve been using Rye a lot recently, and I love its minimalist approach. It seems to "just work" without requiring you to overthink or wrestle with configurations. For a lightweight, efficient Python workflow, Rye is hard to beat.

## 3.1 Installing Rye
[Installing Rye](https://rye.astral.sh/guide/installation/) is straightforward:

```bash
curl -sSf https://rye.astral.sh/get | bash
```
This script installs Rye globally on your system. Once done, ensure the Rye binary is in your PATH. You can verify your installation with:

```bash
rye --version
```

And that’s it—Rye is ready to use!

## 3.2 Creating an Environment with Rye

Rye automatically manages environments for you at the project level, eliminating the need to manually create or activate environments. You can create a new project with Rye:

```bash
rye init myproject
```

This command will create a new directory called myproject and populate it with files
and folders:

```

├── pyproject.toml
├── README.md
└── src
    └── myproject
        └── __init__.py
```

If you already have a project, you can `cd` in your project folder and use

```bash
rye init .
```

This command will fail if you already have a `pyproject.toml` file. You could rename your file to `pyproject.toml.bak` to keep it as a backup if needed! 

What’s unique about Rye is that it automatically uses the correct environment whenever you run commands inside the project directory—no need to manually activate or deactivate environments. If you’re curious about the current environment, you can run:

```bash
rye show
```

This will display details about the environment and its dependencies, for example:

```
project: myproject
path: /home/username/Documents/.private/myproject
venv: /home/username/Documents/.private/myproject/.venv
target python: 3.12
venv python: cpython@3.12.3
virtual: false
configured sources:
  default (index: https://pypi.org/simple/)
```


## 3.3 Installing Dependencies with Rye
Adding dependencies with Rye is as simple as running:

bash
Copy code
rye add numpy pandas
Need a specific version? Just specify it:

bash
Copy code
rye add numpy==1.24 pandas>=2.0
Rye automatically updates the pyproject.toml file, ensuring all dependencies are tracked. This makes your environment reproducible and easy to share.

For development-only dependencies (like linters or test tools), use the --dev flag:

bash
Copy code
rye add --dev pytest flake8
To remove a dependency, run:

bash
Copy code
rye remove numpy
Rye will uninstall the package and update the pyproject.toml accordingly.
The pyproject.toml file is Rye’s equivalent of an environment.yaml file in Conda. It tracks all your dependencies, their versions, and configurations. To ensure everyone working on the project has the same setup, you can sync dependencies with:

bash
Copy code
rye sync
This installs all dependencies listed in the pyproject.toml file.

If exact versions are crucial, use:

bash
Copy code
rye pin
This creates a rye.lock file, which locks the specific versions of all dependencies. This ensures consistent environments, even if upstream libraries are updated.
Rye in Action
Let’s summarize Rye’s workflow:

Start a new project: rye init myproject
Add dependencies: rye add numpy pandas
Sync environments: rye sync (for collaborators)
Ensure consistency: rye pin
Rye’s automatic environment handling, seamless integration with pyproject.toml, and built-in dependency management make it a powerful choice for Python projects.

Rye is perfect for Python-only workflows where simplicity and speed matter most. Its opinionated design removes many of the hassles of traditional tools, letting you focus on writing code instead of managing environments. If you’re tired of juggling multiple tools and configurations, Rye is definitely worth trying.




# 4. Miniforge conda or Rye?