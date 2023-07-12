@echo on
set "input=%~1"
set "scale=%~2"
set "input_filename=%~3"

set models[0]=HAT_SRx2
set models[1]=HAT_SRx3
set models[2]=HAT_SRx4
if "%~1"=="" set "input=.\datasets\RealSR"
if "%~2"=="" set "scale=4"
if "%~3"=="" set "input_filename=cavity"
python make_hat_yml.py --input "%input%" --scale "%scale%" > TempVar
set /p "hat=" < TempVar
del TempVar

setlocal EnableDelayedExpansion
echo %hat%

@REM Generate SR output
python .\hat\test.py -opt %hat%

set /a index=%scale%-2
set dir_1=.\results\
set dir_3=\visualization\custom\
set dir_4=%input_filename%_!models[%index%]!.png

@REM Generated SR location
echo !models[%index%]!
echo %dir_1%!models[%index%]!%dir_3%%dir_4%

@REM Get target 16k texture location
set target=.\16k_targets\female_young.exr

@REM TODO: texture transfer

@REM TODO: output result new texture exr file
