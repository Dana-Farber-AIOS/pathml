@echo off

rem showinf.bat: a batch file for displaying information about a given
rem              image file, while displaying it in the image viewer

rem Required JARs: bioformats_package.jar

setlocal
set BF_DIR=%~dp0
if "%BF_DIR:~-1%" == "\" set BF_DIR=%BF_DIR:~0,-1%

set BF_PROG=loci.formats.tools.ImageInfo
call "%BF_DIR%\bf.bat" %*
