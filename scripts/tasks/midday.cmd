@echo off
cd /d C:\Users\jo\orca\workspaces\TradingAgents\sawfish
"C:\Users\jo\.local\bin\claude.exe" -p "Read and follow scripts/prompts/midday_run.md exactly." --permission-mode bypassPermissions >> results\agentic\cron_midday.log 2>&1
