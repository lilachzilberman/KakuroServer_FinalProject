for /f "delims=" %%a in ('cd') do @set GRADLE_USER_HOME=%%a
echo Building with Gradle...
cmd /C gradlew build
echo Copying build output...
copy /y build\libs\ %DEPLOYMENT_TARGET%\webapps\
cd %DEPLOYMENT_TARGET%\webapps\
del ROOT.war
rename *.war ROOT.war
