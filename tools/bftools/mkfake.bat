@echo off

rem mkfake: a script for creating a fake file / directory structures
rem         on the file system

rem Required JARs: bioformats_package.jar

setlocal
set BF_DIR=%~dp0
if "%BF_DIR:~-1%" == "\" set BF_DIR=%BF_DIR:~0,-1%

set BF_PROG=loci.formats.tools.ImageFaker
call "%BF_DIR%\bf.bat" %*
