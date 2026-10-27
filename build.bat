@echo off
cls

echo param arg[0]: %0%
echo param arg[1]: %1%

if "%1%"=="-c" (
	call:clean
	goto :eof
)

if "%1%"=="" (
	call:build Release
) else (
	setlocal enabledelayedexpansion
	set mode=%1%
	if "!mode!"=="Debug" ( call:build !mode! )^
	else if "!mode!"=="Release" ( call:build !mode! )^
	else if "!mode!"=="RelWithDebInfo" ( call:build !mode! )^
	else if "!mode!"=="MinSizeRel" ( call:build !mode! ) else ( call:build_err !mode! )
	endlocal
)

:build_err
echo build failed, mode:[%1%] is invalid...
exit /B 1

:clean
if exist build (
	echo remove build
	rd /s /q build
)
if exist output (
	echo remove output
	rd /s /q output
)
exit /B 0


:build
echo ====== Build Project ======
set mode=%1%
echo Use Build Type: %mode%
if not exist build_win (
	mkdir build_win
)

cd build_win
cmake .. -G "Visual Studio 17 2022" -DCMAKE_CONFIGURATION_TYPES=%mode% -DSAMPLE_BUILD_TARGET="sim"
echo ---^>^>after cmake, err_level:[%errorlevel%]
if %errorlevel% neq 0 ( exit 1 )
msbuild samples.sln -p:Configuration=%mode%
echo ---^>^>after msbuild, err_level:[%errorlevel%]
if %errorlevel% neq 0 ( exit 1 )
cmake -P cmake_install.cmake
echo ---^>^>after install, err_level:[%errorlevel%]
if %errorlevel% neq 0 ( exit 1 )
cd ..\
echo ====== End Build ======
exit 0
