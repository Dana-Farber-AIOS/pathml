@echo off

rem config.bat: master configuration file for the batch files

rem Running this command directly has no effect,
rem but you can tweak the settings to your liking.

rem Set the amount of RAM available to the command line tools.
rem Use "m" suffix for megabytes, "g" for gigabytes; e.g., 2g = 2GB.
rem set BF_MAX_MEM=1g

rem Set the NO_UPDATE_CHECK flag to skip the update check.
rem set NO_UPDATE_CHECK=1

rem If you are behind a proxy server, the host name and port must be set.
rem set PROXY_HOST=
rem set PROXY_PORT=

rem If your CLASSPATH already includes the needed classes,
rem you can set the BF_DEVEL environment variable to
rem disable the required JAR library checks.
rem set BF_DEVEL=1

rem Set the directory containing the JAR libraries.
if "%BF_JAR_DIR%" == "" (
  if exist "%BF_DIR%\..\artifacts" (
    rem Batch files reside in a git working copy.
    rem Look for JARs in the artifacts directory.
    set BF_JAR_DIR=%BF_DIR%\..\artifacts
  ) else (
    rem Batch files reside in a standalone distribution.
    rem Look for JARs in the same directory as the batch files.
    set BF_JAR_DIR=%BF_DIR%
  )
)
