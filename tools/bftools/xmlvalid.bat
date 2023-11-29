@echo off

rem xmlvalid.bat: a batch file for validating XML files

rem Required JARs: bioformats_package.jar

setlocal
set BF_DIR=%~dp0
if "%BF_DIR:~-1%" == "\" set BF_DIR=%BF_DIR:~0,-1%

set BF_PROG=loci.formats.tools.XMLValidate
call "%BF_DIR%\bf.bat" %*
