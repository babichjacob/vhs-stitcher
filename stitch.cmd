@echo off
rem https://stackoverflow.com/a/761658

set arg1=%1

rem throw the first parameter away
shift
set params=%1
:loop
shift
if [%1]==[] goto afterloop
set params=%params% %1
goto loop
:afterloop
@echo on

py -m vhs_stitcher.%arg1% %params%
