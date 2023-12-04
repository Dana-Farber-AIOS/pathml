@echo off

rem bfconvert.bat: a batch file for converting image files between formats

rem Required JARs: bioformats_package.jar

setlocal
set BF_DIR=%~dp0
if "%BF_DIR:~-1%" == "\" set BF_DIR=%BF_DIR:~0,-1%

set BF_PROG=loci.formats.tools.ImageConverter
call "%BF_DIR%\bf.bat" %*
