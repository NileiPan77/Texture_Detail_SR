@echo on
set start=%time%

set "input=%~1"
set "scale=%~2"
set "output=%~3"
set "target=%~4"

if "%~1"=="" set "input=.\data\cavity.exr"
if "%~2"=="" set "scale=4"
if "%~3"=="" set "output=.\16k_targets\cavity_16k.exr"
if "%~4"=="" set "target=.\16k_targets\Height.exr"

@REM Generate upscaled bicubic output
python .\upscale_exr.py --input %input% --scale %scale% --output %output% --method bicubic

@REM TODO: texture transfer and output result new texture exr file
python .\feature_transfer.py --base_cavity %output% --target_height %target% --output_dir .\16k_targets

set end=%time%
echo Total time used: %start% - %end%