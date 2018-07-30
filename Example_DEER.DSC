#DESC	1.2 * DESCRIPTOR INFORMATION ***********************
*
*	Dataset Type and Format:
*
DSRC	EXP
BSEQ	BIG
IKKF	CPLX
XTYP	IDX
YTYP	NODATA
ZTYP	NODATA
*
*	Item Formats:
*
IRFMT	D
IIFMT	D
*
*	Data Ranges and Resolutions:
*
XPTS	252
XMIN	-60.000000
XWID	3514.000000
*
*	Documentational Text:
*
TITL	'ExoU D133R1 S142A G439R1 3500ns'
IRNAM	'Intensity'
IINAM	'Intensity'
XNAM	'Time'
IRUNI	''
IIUNI	''
XUNI	'ns'
*
************************************************************
*
#SPL	1.2 * STANDARD PARAMETER LAYER
*
OPER    xepr
DATE    05/29/18
TIME    15:38:25
CMNT    
SAMP    
SFOR    
STAG    C
EXPT    
OXS1    
AXS1    
AXS2    
AXS3    
MWPW    0.05
A1CT    1.1995
B0VL    1.1995
A1SW    0.02
MWFQ    3.369549e+10
AVGS    191
*
************************************************************
*
#DSL	1.0 * DEVICE SPECIFIC LAYER
*

.DVC     acqStart, 1.0


.DVC     cwBridge, 1.0

AcqFineTuning      Never
AcqScanFTuning     Off
AcqSliceFTuning    Off
Power              50.00 mW
PowerAtten         0 dB
QValue             600

.DVC     fieldCtrl, 1.0

AllegroMode        False
CenterField        11995.00 G
Delay              0.0 s
FieldFlyback       On
FieldPosition      11995.000 G
FieldWait          Wait LED off
GFactor            2.000000
MeasuringHall      False
SetToSampleG       False
StaticFieldMon     12140.000 G
SweepDirection     Up
SweepWidth         200.0 G
WidthTM            200.0 G

.DVC     freqCounter, 1.0

FrequencyMon       33.695492 GHz
QMonitBridge       On

.DVC     ftBridge, 1.0

Attenuation        60.00 dB
ELDORAtt           0 dB
FrequencyA         33.748700 GHz
VideoBW            200 MHz
VideoGain          33 dB

.DVC     ftEpr, 1.0

AWGPhaseShift      90.00 deg.
AutoTimeOut        True
AveragesPerScan    100
FTEzAWGELDORa      100.00 %
FTEzAWGELDORf      0.000 MHz
FTEzAWGELDORw      0.000 MHz
FTEzB0SettlDel     0 s
FTEzB0Step         0 G
FTEzCurrELDOR      33.748700 GHz
FTEzD11deer        160 ns
FTEzD20deer        3500 ns
FTEzD3deer         100 ns
FTEzDXdeer         14 ns
FTEzDelay0         770 ns
FTEzDelay1         0 ns
FTEzDelay2         0 ns
FTEzDelay4         0 ns
FTEzDeltaX         0 ns
FTEzDeltaY         0 ns
FTEzELDORStart     33.240333 GHz
FTEzELDORWidth     0.000000 GHz
FTEzENDORDetDel    0 ns
FTEzExtTrgDel      0 ns
FTEzIntgBase       1 ns
FTEzIntgGate       80 ns
FTEzMWPiHalf       16 ns
FTEzNPls           1
FTEzP20deer        32 ns
FTEzP3             0 ns
FTEzRF1Pi          0 ns
FTEzRF1PlsSep      2 ns
FTEzRF1Pump        0.500 MHz
FTEzRF1Start       0.500 MHz
FTEzRF1Width       0.000 MHz
FTEzSRELDORExp     False
FTEzSXdeer         252
FTEzSXdeerExp      True
FTEzSleep          0 s
FieldIsStatic      False
GradIntPulse       False
GrdEnable          False
LastXAxis          ?
LastYAxis          ?
MDigPrg            {3;31,1,1;1[mixed]} 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1
MMWaveLOFreq       26.880000 GHz
OnlyAWGChans       False
PCycleAllowed      True
PCycleOn           True
PPExtTrg           False
PPExtTrgSlope      Rising
PlsSPELEXPSlct     4P-DEER
PlsSPELLISTSlct    2-step
PlsSPELPhPrgEx     Normal
QuadDetect         True
ReplaceMode        Off
ShotRepTime        849.66 us
ShotsPLoop         100
SmoothAllowed      False
SmoothPoints       1
SptProgress        53 %
StochMode          False
SweepsPExp         2000
TriggerTimeOut     57 s
XSpecRes           252
YSpecRes           1

.DVC     recorder, 1.0

BaselineCorr       Off
NbScansAcc         191
NbScansDone        191
NbScansToDo        2000
ReplaceMode        Off
SmoothMode         Auto
SmoothPoints       1

.DVC     routeTrg, 1.0


.DVC     sigChanSmall, 1.0


.DVC     transRec, 1.0


*
************************************************************
