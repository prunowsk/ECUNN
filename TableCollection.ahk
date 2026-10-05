#Requires AutoHotkey v2.0
#SingleInstance force
#Warn
SendMode "Event"

global function := Functions()

;Force Quit button
F1:: {
	Suspend
	ToolTip "Exiting"
	Sleep 500
	ToolTip()
	SetMouseDelay 0
	ExitApp
	
}

;Force Restart button
F2::{
	ToolTip "Reloading"
	Sleep 500
	ToolTip()
	SetMouseDelay 0
	Reload
	
}

;Main button
F3::{
	ToolTip "Starting"
	Sleep 500
	ToolTip()
	
	function.CollectData()
	
	
}


class Functions{

    ;Adds data to file in script directory CarInputInfo.txt
	CollectData(){
		filePath := A_ScriptDir "\CarInputInfo.txt"
		x := 40
		y := 1
		s := 30
		
		MouseMove 50,115
		SendEvent "{click left down}"
		MouseMove 355,500
		SendEvent "{click left up}"

		Loop 7 {
			if (y != 5){
				MouseMove 480,40
				SendEvent "{click left}"
				Sleep s
				y := y + 1
				x := x + 15
				MouseMove 480, x
				SendEvent "{click left}"
				Sleep s
				send ("^c")
				Sleep s
				FileAppend(A_Clipboard "`n",filePath)
				Sleep s
			}
			else {
				y := y + 1
				x := x + 15
			}
			
		}
		
	}
	



}