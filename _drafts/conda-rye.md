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
        <li><a href="#4-miniforge-or-rye">4. Miniforge or Rye?</a></li>
    </ul>
</div>

# 1. Introduction

Managing Python environments and dependencies is crucial for smooth workflows and avoiding compatibility headaches. Each project often needs specific Python versions or libraries, and manually managing them can get messy fast. Tools designed for this purpose help by isolating environments, ensuring each project has what it needs without interfering with others. This keeps things consistent, collaborative, and reliable.

With so many tools out there—venv, virtualenv, Conda, pyenv, Poetry, ...—it’s easy to feel overwhelmed. In this post, I’ll introduce two tools that, in my view, strike a great balance between simplicity and functionality: Miniforge and Rye. I will assume that you're on a UNIX-based computer (Linux or MacOS), but Windows users can follow this post using WSL.


# 2. Miniforge conda

[Miniforge](https://github.com/conda-forge/miniforge) is a lightweight, open-source implementation of ``conda`` (and `mamba`) specific to conda-forge. The packages in the base environments are obtained from [conda-forge](https://conda-forge.org/) (open-source repositories of conda packages), and the conda-forge channel is set as the default channel. In a nutshell, Miniforge is the fully open-source community-driven alternative to Anaconda.

 

<div class="callout callout-tip">
  <div>
  While Anaconda remains a popular choice, recent updates to its licensing terms have made it less ideal for many users. As of March 2024, organizations with 200 or more employees must obtain a paid license to use Anaconda’s software, including its default package channels. This shift has impacted not only commercial companies but also academic and research institutions that previously assumed they could use Anaconda for free (e.g., see this <a href="https://web.archive.org/web/20241202191039/https://www.rc.virginia.edu/2024/10/transition-from-anaconda-to-miniforge-october-15-2024/">post</a> from the University of Virginia). Given these changes, I recommend Miniforge as a more open-source-friendly, lightweight, and license-free alternative.
  </div>
</div>

## 2.1 Installing Miniforge conda

The installation process is quite simple. As explained in their github [repo](https://github.com/conda-forge/miniforge), you can 
download and execute the installation script using
```bash
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
```

You can follow the installation process (you'll be prompted several times). At the end, miniforge will ask if you _"wish to update your shell profile to automatically initialize conda?"_ : you can answer yes. It will add in your `.bashrc` (or `.zshrc` is you're using the zsh shell) the following lines
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

Note that you shouldn't modify anything written in this block, it is fully managed by miniforge.



If you restart your shell, two things will occur: (1) you have now access to the `conda` executable and (2) the base environment is automatically activated when you start a new shell. I would strongly recommend deactivating this auto-activation behavior by using 

```bash
conda config --set auto_activate_base false
```

Then restarting your shell (last time, I promise!). Now everything should be ready! You can check your installation using the command `conda info`. Here are the first fews lines of the expected output

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

Once Miniforge `conda` is installed, the next step is to create isolated environments for your projects. Environments ensure that each project has its own set of dependencies and Python version, avoiding conflicts between projects. To create a new environment, use the conda create command. For example, to create an environment named myenv with Python 3.10, run:

```bash
conda create -n myenv python=3.12
```

* `-n myenv` specifies the name of the environment (myenv).
* `python=3.12` ensures the environment uses Python 3.12.


To start working in the environment you just created, activate it with:

```bash
conda activate myenv
```

Once activated, the shell prompt will show the environment’s name, indicating you’re working within that environment. You can also use ``conda env list`` to list all the available environments, along with their paths.

<div class="callout callout-info">
  <div>
    When you activate an environment with <code>conda activate myenv</code>:
    <ul>
      <li>
        The environment’s path is moved to the front of your PATH, ensuring commands like 
        <code>python</code> and <code>pip</code> use the environment-specific versions.
      </li>
      <li>
        Environment variables like <code>CONDA_PREFIX</code> are set to help programs find 
        the right dependencies.
      </li>
      <li>
        Your shell prompt updates (e.g., <code>(myenv)</code>), so you know which 
        environment is active.
      </li>
      <li>
        Everything you do—installing packages, running scripts—stays isolated to that 
        environment.
      </li>
    </ul>
  </div>
</div>


When you’re done, you can deactivate the environment with:

```bash
conda deactivate
```

If an environment is no longer needed, you can delete it with:

```bash
conda remove -n myenv --all
```

The ``--all`` flag ensures everything associated with the environment is removed.


## 2.3 Installing Dependencies with Miniforge conda
With your environment set up, you can start adding dependencies. Miniforge uses conda-forge as the default channel, so all installed packages are open-source and community-maintained. First, activate your environmnent, e.g., `conda activate myenv`. To install packages, use the `conda install` command, e.g., 

``bash
Copy code
conda install numpy pandas>=2.0 matplotlib
```
Specifying Package Versions
If you need a specific version of a package, include it in the command. For instance:

bash
Copy code
conda install numpy=1.24
Updating a Package
To update a package to the latest version:

bash
Copy code
conda update numpy
Removing a Package
If a package is no longer needed, you can remove it with:

bash
Copy code
conda remove numpy
Exporting and Reproducing Environments
To share or reproduce an environment, you can export it to a file:

bash
Copy code
conda env export > environment.yml
This creates a yml file listing all dependencies and versions. Others can recreate the environment using:

bash
Copy code
conda env create -f environment.yml
By mastering these commands, you’ll be able to efficiently manage environments and dependencies with Miniforge, keeping your Python projects clean, organized, and reproducible.

# 3. Rye

## 3.1 Installing Rye

## 3.2 Creating an Environment with Rye

## 3.3 Installing Dependencies with Rye

# 4. Miniforge or Rye?