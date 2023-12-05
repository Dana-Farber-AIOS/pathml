@echo off

rem bf-unconfigured.bat: a batch file for identifying datasets with no .bioformats configuration

rem Required JARs: bioformats_package.jar, bio-formats-testing-framework.jar

setlocal
set BF_DIR=%~dp0
if "%BF_DIR:~-1%" == "\" set BF_DIR=%BF_DIR:~0,-1%

set BF_PROG=loci.tests.testng.ReportEnabledStatus
call "%BF_DIR%\bf.bat" %*
