# WooldridgeCode.jl

Julia code for ["Introductory Econometrics" A Modern Approach", Seventh Edition](https://www.cengage.uk/c/introductory-econometrics-a-modern-approach-7e-wooldridge/9781337558860/) by Jeffrey M. Wooldridge

* The code cover *Part 1: "Regression Analysis with Cross-Sectional Data"* of the book (chapters 1 to 9).
* The code cover all examples, problems, and computer problems given in the book.
* In the code the GLM.jl package is used.
* The *init_example.jl* file contains helper functions that were needed to make the code complete
  (getting data and functionalities currently missing in GLM.jl).
* For each chapter a file called *chapter_XX.jl* is provided, where *XX* is chapter number.

The code were tested under Julia 1.9.2 and Project.toml/Manifest.toml configuration provided in this repository.

When you run the code please consider that:

* A project environment is properly activated in your Julia session
  (it is simplest to do it by running `julia --project` in the folder where you downloaded this repository).
* The code downloads source R files from https://github.com/JustinMShea/wooldridge and stores them in a working directory.
* It is assumed that code for each chapter is executed in a separate Julia session sequentially
  (i.e. there might be some variables defined in the code for a given chapter that are used in several places)
