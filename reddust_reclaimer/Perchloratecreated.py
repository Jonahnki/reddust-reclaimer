import cPickle, base64

try:
    from SimpleSession.versions.v65 import (
        beginRestore,
        registerAfterModelsCB,
        reportRestoreError,
        checkVersion,
    )
except ImportError:
    from chimera import UserError

    raise UserError(
        "Cannot open session that was saved in a"
        " newer version of Chimera; update your version"
    )
checkVersion([1, 19, 42556])
import chimera
from chimera import replyobj

replyobj.status("Restoring session...", blankAfter=0)
replyobj.status("Beginning session restore...", blankAfter=0, secondary=True)
beginRestore()


def restoreCoreModels():
    from SimpleSession.versions.v65 import (
        init,
        restoreViewer,
        restoreMolecules,
        restoreColors,
        restoreSurfaces,
        restoreVRML,
        restorePseudoBondGroups,
        restoreModelAssociations,
    )

    molInfo = cPickle.loads(
        base64.b64decode(
            "gAJ9cQEoVRFyaWJib25JbnNpZGVDb2xvcnECSwNOfYdVCWJhbGxTY2FsZXEDSwNHP9AAAAAAAAB9h1UJcG9pbnRTaXplcQRLA0c/8AAAAAAAAH2HVQVjb2xvcnEFSwNLAH1xBihLAV1xB0sBYUsCXXEISwJhdYdVCnJpYmJvblR5cGVxCUsDSwB9h1UKc3RpY2tTY2FsZXEKSwNHP/AAAAAAAAB9h1UMbW1DSUZIZWFkZXJzcQtdcQwoTk5OZVUMYXJvbWF0aWNNb2RlcQ1LA0sBfYdVCnZkd0RlbnNpdHlxDksDR0AUAAAAAAAAfYdVBmhpZGRlbnEPSwOJfYdVDWFyb21hdGljQ29sb3JxEEsDTn2HVQ9yaWJib25TbW9vdGhpbmdxEUsDSwB9h1UJYXV0b2NoYWlucRJLA4h9h1UKcGRiVmVyc2lvbnETSwNLAH2HVQhvcHRpb25hbHEUfXEVVQtjaGFyZ2VNb2RlbHEWiIlLA059cRdVDEFNQkVSIGZmMTRTQnEYXXEZSwBhc4eHc1UPbG93ZXJDYXNlQ2hhaW5zcRpLA4l9h1UJbGluZVdpZHRocRtLA0c/8AAAAAAAAH2HVQ9yZXNpZHVlTGFiZWxQb3NxHEsDSwB9h1UEbmFtZXEdSwNYBwAAAHNjcmF0Y2h9h1UPYXJvbWF0aWNEaXNwbGF5cR5LA4l9h1UPcmliYm9uU3RpZmZuZXNzcR9LA0c/6ZmZmZmZmn2HVQpwZGJIZWFkZXJzcSBdcSEofXEifXEjfXEkZVUDaWRzcSVLA0sCSwCGfXEmKEsBSwCGXXEnSwFhSwBLAIZdcShLAGF1h1UOc3VyZmFjZU9wYWNpdHlxKUsDR7/wAAAAAAAAfYdVEGFyb21hdGljTGluZVR5cGVxKksDSwJ9h1UUcmliYm9uSGlkZXNNYWluY2hhaW5xK0sDiH2HVQdkaXNwbGF5cSxLA4h9h3Uu"
        )
    )
    resInfo = cPickle.loads(
        base64.b64decode(
            "gAJ9cQEoVQZpbnNlcnRxAksDVQEgfYdVC2ZpbGxEaXNwbGF5cQNLA4l9h1UEbmFtZXEESwNYAgAAAENsfXEFWAEAAABPXXEGSwJhc4dVBWNoYWlucQdLA1gDAAAAaGV0fYdVDnJpYmJvbkRyYXdNb2RlcQhLA0sCfYdVAnNzcQlLA4mJhn2HVQhtb2xlY3VsZXEKSwNLAH1xCyhLAU5dcQxLAUsBhnENYYZLAk5dcQ5LAksBhnEPYYZ1h1ULcmliYm9uQ29sb3JxEEsDTn2HVQVsYWJlbHERSwNYAAAAAH2HVQpsYWJlbENvbG9ycRJLA059h1UIZmlsbE1vZGVxE0sDSwF9h1UFaXNIZXRxFEsDiH2HVQtsYWJlbE9mZnNldHEVSwNOfYdVCHBvc2l0aW9ucRZdcRcoSwFLAYZxGEsBSwGGcRlLAUsBhnEaZVUNcmliYm9uRGlzcGxheXEbSwOJfYdVCG9wdGlvbmFscRx9VQRzc0lkcR1LA0r/////fYd1Lg=="
        )
    )
    atomInfo = cPickle.loads(
        base64.b64decode(
            "gAJ9cQEoVQdyZXNpZHVlcQJLB0sFfXEDKEsDTl1xBEsASwGGcQVhhksETl1xBksBSwGGcQdhhnWHVQh2ZHdDb2xvcnEISwdLBX1xCShLA11xCihLAEsBZUsEXXELSwJhdYdVBG5hbWVxDEsHWAMAAABIZTF9cQ0oWAMAAABIMTFdcQ5LA2FYAwAAAEgxMl1xD0sEYVgCAAAASDFdcRBLBmFYBAAAAEhPMTFdcRFLBWFYAgAAAE8xXXESSwJhdYdVA3Zkd3ETSweJfYdVDnN1cmZhY2VEaXNwbGF5cRRLB4l9h1UFY29sb3JxFUsHSwV9cRYoSwNdcRcoSwBLAWVLBF1xGEsCYXWHVQlpZGF0bVR5cGVxGUsHiX2HVQZhbHRMb2NxGksHVQB9h1UFbGFiZWxxG0sHWAAAAAB9h1UOc3VyZmFjZU9wYWNpdHlxHEsHR7/wAAAAAAAAfYdVB2VsZW1lbnRxHUsHSwF9cR4oSwhdcR9LAmFLAl1xIChLAEsBZXWHVQpsYWJlbENvbG9ycSFLB0sFfXEiKEsDXXEjKEsASwFlSwRdcSRLAmF1h1UMc3VyZmFjZUNvbG9ycSVLB0sFfXEmKEsDXXEnKEsASwFlSwRdcShLAmF1h1UPc3VyZmFjZUNhdGVnb3J5cSlLB1gEAAAAbWFpbn1xKlgHAAAAc29sdmVudE5dcStLAEsChnEsYYZzh1UGcmFkaXVzcS1LB0c/8AAAAAAAAH1xLihHP/zMzMAAAABdcS8oSwBLAWVHP/a4UeAAAABdcTBLAmF1h1UKY29vcmRJbmRleHExXXEyKEsASwGGcTNLAEsBhnE0SwBLBYZxNWVVC2xhYmVsT2Zmc2V0cTZLB059h1USbWluaW11bUxhYmVsUmFkaXVzcTdLB0cAAAAAAAAAAH2HVQhkcmF3TW9kZXE4SwdLAn2HVQhvcHRpb25hbHE5fXE6KFUGY2hhcmdlcTuIiUsHTn1xPEr/////XXE9SwBhc4eHVQxzZXJpYWxOdW1iZXJxPoiIXXE/KEsASwGGcUBLAEsBhnFBSwBLBYZxQmWHVQdiZmFjdG9ycUOIiUsHRwAAAAAAAAAAfYeHVQlvY2N1cGFuY3lxRIiJSwdHP/AAAAAAAAB9h4d1VQdkaXNwbGF5cUVLB4h9h3Uu"
        )
    )
    bondInfo = cPickle.loads(
        base64.b64decode(
            "gAJ9cQEoVQVjb2xvcnECSwROfYdVBWF0b21zcQNdcQQoXXEFKEsJSwhlXXEGKEsKSwhlXXEHKEsLSwhlXXEIKEsMSwhlZVUFbGFiZWxxCUsEWAAAAAB9h1UIaGFsZmJvbmRxCksEiH2HVQZyYWRpdXNxC0sERz/JmZmgAAAAfYdVC2xhYmVsT2Zmc2V0cQxLBE59h1UIZHJhd01vZGVxDUsESwF9h1UIb3B0aW9uYWxxDn1VB2Rpc3BsYXlxD0sESwJ9h3Uu"
        )
    )
    crdInfo = cPickle.loads(
        base64.b64decode(
            "gAJ9cQEoSwB9cQIoVQZhY3RpdmVxA0sBSwFdcQRHAAAAAAAAAABHAAAAAAAAAABHAAAAAAAAAACHcQVhdUsBfXEGKGgDSwFLAV1xB0cAAAAAAAAAAEcAAAAAAAAAAEcAAAAAAAAAAIdxCGF1SwJ9cQkoaANLAUsBXXEKKEcAAAAAAAAAAEcAAAAAAAAAAEcAAAAAAAAAAIdxC0c/7qFh5Pdl/kcAAAAAAAAAAEcAAAAAAAAAAIdxDEe/1HMDs8ft00c/7N+cxhl/MUcAAAAAAAAAAIdxDUe/1GfgCxNvFEe/3N+cxhl/Mke/6QOZRZESeIdxDke/1GuWbWsP7Ue/3OTdiUqTOEc/6QFTE2uXI4dxD2V1dS4="
        )
    )
    surfInfo = {
        "category": (0, None, {}),
        "probeRadius": (0, None, {}),
        "pointSize": (0, None, {}),
        "name": [],
        "density": (0, None, {}),
        "colorMode": (0, None, {}),
        "useLighting": (0, None, {}),
        "transparencyBlendMode": (0, None, {}),
        "molecule": [],
        "smoothLines": (0, None, {}),
        "lineWidth": (0, None, {}),
        "allComponents": (0, None, {}),
        "twoSidedLighting": (0, None, {}),
        "customVisibility": [],
        "drawMode": (0, None, {}),
        "display": (0, None, {}),
        "customColors": [],
    }
    vrmlInfo = {
        "subid": (0, None, {}),
        "display": (0, None, {}),
        "id": (0, None, {}),
        "vrmlString": [],
        "name": (0, None, {}),
    }
    colors = {
        "Ru": ((0.141176, 0.560784, 0.560784), 1, "default"),
        "Re": ((0.14902, 0.490196, 0.670588), 1, "default"),
        "Rf": ((0.8, 0, 0.34902), 1, "default"),
        "Ra": ((0, 0.490196, 0), 1, "default"),
        "Rb": ((0.439216, 0.180392, 0.690196), 1, "default"),
        "Rn": ((0.258824, 0.509804, 0.588235), 1, "default"),
        "Rh": ((0.0392157, 0.490196, 0.54902), 1, "default"),
        "Be": ((0.760784, 1, 0), 1, "default"),
        "Ba": ((0, 0.788235, 0), 1, "default"),
        "Bh": ((0.878431, 0, 0.219608), 1, "default"),
        "Bi": ((0.619608, 0.309804, 0.709804), 1, "default"),
        "Bk": ((0.541176, 0.309804, 0.890196), 1, "default"),
        "Br": ((0.65098, 0.160784, 0.160784), 1, "default"),
        "H": ((1, 1, 1), 1, "default"),
        "P": ((1, 0.501961, 0), 1, "default"),
        "Os": ((0.14902, 0.4, 0.588235), 1, "default"),
        "Es": ((0.701961, 0.121569, 0.831373), 1, "default"),
        "Hg": ((0.721569, 0.721569, 0.815686), 1, "default"),
        "Ge": ((0.4, 0.560784, 0.560784), 1, "default"),
        "Gd": ((0.270588, 1, 0.780392), 1, "default"),
        "Ga": ((0.760784, 0.560784, 0.560784), 1, "default"),
        "Pr": ((0.85098, 1, 0.780392), 1, "default"),
        "Pt": ((0.815686, 0.815686, 0.878431), 1, "default"),
        "Pu": ((0, 0.419608, 1), 1, "default"),
        "C": ((0.564706, 0.564706, 0.564706), 1, "default"),
        "Pb": ((0.341176, 0.34902, 0.380392), 1, "default"),
        "Pa": ((0, 0.631373, 1), 1, "default"),
        "Pd": ((0, 0.411765, 0.521569), 1, "default"),
        "Cd": ((1, 0.85098, 0.560784), 1, "default"),
        "Po": ((0.670588, 0.360784, 0), 1, "default"),
        "Pm": ((0.639216, 1, 0.780392), 1, "default"),
        "Hs": ((0.901961, 0, 0.180392), 1, "default"),
        "Ho": ((0, 1, 0.611765), 1, "default"),
        "Hf": ((0.301961, 0.760784, 1), 1, "default"),
        "K": ((0.560784, 0.25098, 0.831373), 1, "default"),
        "He": ((0.85098, 1, 1), 1, "default"),
        "Md": ((0.701961, 0.0509804, 0.65098), 1, "default"),
        "Mg": ((0.541176, 1, 0), 1, "default"),
        "Mo": ((0.329412, 0.709804, 0.709804), 1, "default"),
        "Mn": ((0.611765, 0.478431, 0.780392), 1, "default"),
        "O": ((1, 0.0509804, 0.0509804), 1, "default"),
        "Mt": ((0.921569, 0, 0.14902), 1, "default"),
        "S": ((1, 1, 0.188235), 1, "default"),
        "W": ((0.129412, 0.580392, 0.839216), 1, "default"),
        "sky blue": ((0.529412, 0.807843, 0.921569), 1, "default"),
        "Zn": ((0.490196, 0.501961, 0.690196), 1, "default"),
        "plum": ((0.866667, 0.627451, 0.866667), 1, "default"),
        "Eu": ((0.380392, 1, 0.780392), 1, "default"),
        "Zr": ((0.580392, 0.878431, 0.878431), 1, "default"),
        "Er": ((0, 0.901961, 0.458824), 1, "default"),
        "Ni": ((0.313725, 0.815686, 0.313725), 1, "default"),
        "No": ((0.741176, 0.0509804, 0.529412), 1, "default"),
        "Na": ((0.670588, 0.360784, 0.94902), 1, "default"),
        "Nb": ((0.45098, 0.760784, 0.788235), 1, "default"),
        "Nd": ((0.780392, 1, 0.780392), 1, "default"),
        "Ne": ((0.701961, 0.890196, 0.960784), 1, "default"),
        "Np": ((0, 0.501961, 1), 1, "default"),
        "Fr": ((0.258824, 0, 0.4), 1, "default"),
        "Fe": ((0.878431, 0.4, 0.2), 1, "default"),
        "Fm": ((0.701961, 0.121569, 0.729412), 1, "default"),
        "B": ((1, 0.709804, 0.709804), 1, "default"),
        "F": ((0.564706, 0.878431, 0.313725), 1, "default"),
        "Sr": ((0, 1, 0), 1, "default"),
        "N": ((0.188235, 0.313725, 0.972549), 1, "default"),
        "Kr": ((0.360784, 0.721569, 0.819608), 1, "default"),
        "Si": ((0.941176, 0.784314, 0.627451), 1, "default"),
        "Sn": ((0.4, 0.501961, 0.501961), 1, "default"),
        "Sm": ((0.560784, 1, 0.780392), 1, "default"),
        "V": ((0.65098, 0.65098, 0.670588), 1, "default"),
        "Sc": ((0.901961, 0.901961, 0.901961), 1, "default"),
        "Sb": ((0.619608, 0.388235, 0.709804), 1, "default"),
        "Sg": ((0.85098, 0, 0.270588), 1, "default"),
        "Se": ((1, 0.631373, 0), 1, "default"),
        "Co": ((0.941176, 0.564706, 0.627451), 1, "default"),
        "Cm": ((0.470588, 0.360784, 0.890196), 1, "default"),
        "Cl": ((0.121569, 0.941176, 0.121569), 1, "default"),
        "Ca": ((0.239216, 1, 0), 1, "default"),
        "Cf": ((0.631373, 0.211765, 0.831373), 1, "default"),
        "Ce": ((1, 1, 0.780392), 1, "default"),
        "Xe": ((0.258824, 0.619608, 0.690196), 1, "default"),
        "Lu": ((0, 0.670588, 0.141176), 1, "default"),
        "Cs": ((0.341176, 0.0901961, 0.560784), 1, "default"),
        "Cr": ((0.541176, 0.6, 0.780392), 1, "default"),
        "Cu": ((0.784314, 0.501961, 0.2), 1, "default"),
        "La": ((0.439216, 0.831373, 1), 1, "default"),
        "Li": ((0.8, 0.501961, 1), 1, "default"),
        "Tl": ((0.65098, 0.329412, 0.301961), 1, "default"),
        "Tm": ((0, 0.831373, 0.321569), 1, "default"),
        "Lr": ((0.780392, 0, 0.4), 1, "default"),
        "Th": ((0, 0.729412, 1), 1, "default"),
        "Ti": ((0.74902, 0.760784, 0.780392), 1, "default"),
        "tan": ((0.823529, 0.705882, 0.54902), 1, "default"),
        "Te": ((0.831373, 0.478431, 0), 1, "default"),
        "Tb": ((0.188235, 1, 0.780392), 1, "default"),
        "Tc": ((0.231373, 0.619608, 0.619608), 1, "default"),
        "Ta": ((0.301961, 0.65098, 1), 1, "default"),
        "Yb": ((0, 0.74902, 0.219608), 1, "default"),
        "Db": ((0.819608, 0, 0.309804), 1, "default"),
        "Dy": ((0.121569, 1, 0.780392), 1, "default"),
        "I": ((0.580392, 0, 0.580392), 1, "default"),
        "U": ((0, 0.560784, 1), 1, "default"),
        "Y": ((0.580392, 1, 1), 1, "default"),
        "Ac": ((0.439216, 0.670588, 0.980392), 1, "default"),
        "Ag": ((0.752941, 0.752941, 0.752941), 1, "default"),
        "Ir": ((0.0901961, 0.329412, 0.529412), 1, "default"),
        "Am": ((0.329412, 0.360784, 0.94902), 1, "default"),
        "Al": ((0.74902, 0.65098, 0.65098), 1, "default"),
        "As": ((0.741176, 0.501961, 0.890196), 1, "default"),
        "Ar": ((0.501961, 0.819608, 0.890196), 1, "default"),
        "Au": ((1, 0.819608, 0.137255), 1, "default"),
        "At": ((0.458824, 0.309804, 0.270588), 1, "default"),
        "In": ((0.65098, 0.458824, 0.45098), 1, "default"),
    }
    materials = {"default": ((0.85, 0.85, 0.85), 30)}
    pbInfo = {
        "category": ["distance monitor"],
        "bondInfo": [
            {
                "color": (0, None, {}),
                "atoms": [],
                "label": (0, None, {}),
                "halfbond": (0, None, {}),
                "labelColor": (0, None, {}),
                "labelOffset": (0, None, {}),
                "drawMode": (0, None, {}),
                "display": (0, None, {}),
            }
        ],
        "lineType": (1, 2, {}),
        "color": (1, 6, {}),
        "optional": {"fixedLabels": (True, False, (1, False, {}))},
        "display": (1, True, {}),
        "showStubBonds": (1, False, {}),
        "lineWidth": (1, 1, {}),
        "stickScale": (1, 1, {}),
        "id": [-2],
    }
    modelAssociations = {}
    colorInfo = (
        8,
        ("H", (1, 1, 1, 1)),
        {
            ("green", (0, 1, 0, 1)): [7],
            ("O", (1, 0.0509804, 0.0509804, 1)): [4],
            ("sky blue", (0.529412, 0.807843, 0.921569, 1)): [1],
            ("He", (0.85098, 1, 1, 1)): [3],
            ("tan", (0.823529, 0.705882, 0.54902, 1)): [0],
            ("plum", (0.866667, 0.627451, 0.866667, 1)): [2],
            ("yellow", (1, 1, 0, 1)): [6],
        },
    )
    viewerInfo = {
        "cameraAttrs": {
            "center": (0, 0, 0),
            "fieldOfView": 7.9350144292525,
            "nearFar": (2.5034287171534, -2.4917623443721),
            "ortho": False,
            "eyeSeparation": 50.8,
            "focal": 0,
        },
        "viewerAttrs": {
            "silhouetteColor": None,
            "clipping": False,
            "showSilhouette": False,
            "showShadows": False,
            "viewSize": 5.8065636083733,
            "labelsOnTop": True,
            "depthCueRange": (0.5, 1),
            "silhouetteWidth": 2,
            "singleLayerTransparency": True,
            "shadowTextureSize": 2048,
            "backgroundImage": [None, 1, 2, 1, 0, 0],
            "backgroundGradient": [
                ("Chimera default", [(1, 1, 1, 1), (0, 0, 1, 1)], 1),
                1,
                0,
                0,
            ],
            "depthCue": True,
            "highlight": 0,
            "scaleFactor": 1,
            "angleDependentTransparency": True,
            "backgroundMethod": 0,
        },
        "viewerHL": 7,
        "cameraMode": "mono",
        "detail": 1.5,
        "viewerFog": None,
        "viewerBG": None,
    }

    replyobj.status("Initializing session restore...", blankAfter=0, secondary=True)
    from SimpleSession.versions.v65 import expandSummary

    init(dict(enumerate(expandSummary(colorInfo))))
    replyobj.status("Restoring colors...", blankAfter=0, secondary=True)
    restoreColors(colors, materials)
    replyobj.status("Restoring molecules...", blankAfter=0, secondary=True)
    restoreMolecules(molInfo, resInfo, atomInfo, bondInfo, crdInfo)
    replyobj.status("Restoring surfaces...", blankAfter=0, secondary=True)
    restoreSurfaces(surfInfo)
    replyobj.status("Restoring VRML models...", blankAfter=0, secondary=True)
    restoreVRML(vrmlInfo)
    replyobj.status("Restoring pseudobond groups...", blankAfter=0, secondary=True)
    restorePseudoBondGroups(pbInfo)
    replyobj.status("Restoring model associations...", blankAfter=0, secondary=True)
    restoreModelAssociations(modelAssociations)
    replyobj.status("Restoring camera...", blankAfter=0, secondary=True)
    restoreViewer(viewerInfo)


try:
    restoreCoreModels()
except:
    reportRestoreError("Error restoring core models")

    replyobj.status("Restoring extension info...", blankAfter=0, secondary=True)


try:
    import StructMeasure
    from StructMeasure.DistMonitor import restoreDistances

    registerAfterModelsCB(restoreDistances, 1)
except:
    reportRestoreError("Error restoring distances in session")


def restoreMidasBase():
    formattedPositions = {}
    import Midas

    Midas.restoreMidasBase(formattedPositions)


try:
    restoreMidasBase()
except:
    reportRestoreError("Error restoring Midas base state")


def restoreMidasText():
    from Midas import midas_text

    midas_text.aliases = {}
    midas_text.userSurfCategories = {}


try:
    restoreMidasText()
except:
    reportRestoreError("Error restoring Midas text state")


def restore_volume_data():
    volume_data_state = {
        "class": "Volume_Manager_State",
        "data_and_regions_state": [],
        "version": 2,
    }
    from VolumeViewer import session

    session.restore_volume_data_state(volume_data_state)


try:
    restore_volume_data()
except:
    reportRestoreError("Error restoring volume data")


def restore_cap_attributes():
    cap_attributes = {
        "cap_attributes": [],
        "cap_color": None,
        "cap_offset": 0.01,
        "class": "Caps_State",
        "default_cap_offset": 0.01,
        "mesh_style": False,
        "shown": True,
        "subdivision_factor": 1.0,
        "version": 1,
    }
    import SurfaceCap.session

    SurfaceCap.session.restore_cap_attributes(cap_attributes)


registerAfterModelsCB(restore_cap_attributes)

geomData = {"AxisManager": {}, "CentroidManager": {}, "PlaneManager": {}}

try:
    from StructMeasure.Geometry import geomManager

    geomManager._restoreSession(geomData)
except:
    reportRestoreError("Error restoring geometry objects in session")


def restoreSession_RibbonStyleEditor():
    import SimpleSession
    import RibbonStyleEditor

    userScalings = []
    userXSections = []
    userResidueClasses = []
    residueData = [
        (3, "Chimera default", "rounded", "unknown"),
        (4, "Chimera default", "rounded", "unknown"),
        (5, "Chimera default", "rounded", "unknown"),
    ]
    flags = RibbonStyleEditor.NucleicDefault1
    SimpleSession.registerAfterModelsCB(
        RibbonStyleEditor.restoreState,
        (userScalings, userXSections, userResidueClasses, residueData, flags),
    )


try:
    restoreSession_RibbonStyleEditor()
except:
    reportRestoreError("Error restoring RibbonStyleEditor state")

trPickle = "gAJjQW5pbWF0ZS5UcmFuc2l0aW9ucwpUcmFuc2l0aW9ucwpxASmBcQJ9cQMoVQxjdXN0b21fc2NlbmVxBGNBbmltYXRlLlRyYW5zaXRpb24KVHJhbnNpdGlvbgpxBSmBcQZ9cQcoVQZmcmFtZXNxCEsBVQ1kaXNjcmV0ZUZyYW1lcQlLAVUKcHJvcGVydGllc3EKXXELVQNhbGxxDGFVBG5hbWVxDWgEVQRtb2RlcQ5VBmxpbmVhcnEPdWJVCGtleWZyYW1lcRBoBSmBcRF9cRIoaAhLFGgJSwFoCl1xE2gMYWgNaBBoDmgPdWJVBXNjZW5lcRRoBSmBcRV9cRYoaAhLAWgJSwFoCl1xF2gMYWgNaBRoDmgPdWJ1Yi4="
scPickle = "gAJjQW5pbWF0ZS5TY2VuZXMKU2NlbmVzCnEBKYFxAn1xA1UHbWFwX2lkc3EEfXNiLg=="
kfPickle = (
    "gAJjQW5pbWF0ZS5LZXlmcmFtZXMKS2V5ZnJhbWVzCnEBKYFxAn1xA1UHZW50cmllc3EEXXEFc2Iu"
)


def restoreAnimation():
    "A method to unpickle and restore animation objects"
    # Scenes must be unpickled after restoring transitions, because each
    # scene links to a 'scene' transition. Likewise, keyframes must be
    # unpickled after restoring scenes, because each keyframe links to a scene.
    # The unpickle process is left to the restore* functions, it's
    # important that it doesn't happen prior to calling those functions.
    import SimpleSession
    from Animate.Session import restoreTransitions
    from Animate.Session import restoreScenes
    from Animate.Session import restoreKeyframes

    SimpleSession.registerAfterModelsCB(restoreTransitions, trPickle)
    SimpleSession.registerAfterModelsCB(restoreScenes, scPickle)
    SimpleSession.registerAfterModelsCB(restoreKeyframes, kfPickle)


try:
    restoreAnimation()
except:
    reportRestoreError("Error in Animate.Session")


def restoreLightController():
    import Lighting

    Lighting._setFromParams(
        {
            "ratio": 1.25,
            "brightness": 1.16,
            "material": [30.0, (0.85, 0.85, 0.85), 1.0],
            "back": [
                (0.35740674433659325, 0.6604015517481454, -0.6604015517481455),
                (1.0, 1.0, 1.0),
                0.0,
            ],
            "mode": "two-point",
            "key": [
                (-0.35740674433659325, 0.6604015517481454, 0.6604015517481455),
                (1.0, 1.0, 1.0),
                1.0,
            ],
            "contrast": 0.83,
            "fill": [
                (0.25056280708573153, 0.25056280708573153, 0.9351131265310293),
                (1.0, 1.0, 1.0),
                0.0,
            ],
        }
    )


try:
    restoreLightController()
except:
    reportRestoreError("Error restoring lighting parameters")


try:
    from BuildStructure.gui import _sessionRestore

    _sessionRestore({"mapped": 0})
except:
    reportRestoreError("Failure restoring Build Structure")


def restoreRemainder():
    from SimpleSession.versions.v65 import (
        restoreWindowSize,
        restoreOpenStates,
        restoreSelections,
        restoreFontInfo,
        restoreOpenModelsAttrs,
        restoreModelClip,
        restoreSilhouettes,
    )

    curSelIds = [8]
    savedSels = []
    openModelsAttrs = {"cofrMethod": 4}
    windowSize = (1600, 777)
    xformMap = {
        0: (
            ((-0.9426211923762, 0.30664527681507, -0.13203772904092), 131.03345216307),
            (0.033908918827039, 0.12222318536463, 0.041775158149793),
            True,
        ),
        1: (
            ((-0.9426211923762, 0.30664527681507, -0.13203772904092), 131.03345216307),
            (0.033908918827039, 0.12222318536463, 0.041775158149793),
            True,
        ),
        2: (
            ((-0.9426211923762, 0.30664527681507, -0.13203772904092), 131.03345216307),
            (0.033908918827039, 0.12222318536463, 0.041775158149793),
            True,
        ),
    }
    fontInfo = {"face": ("Sans Serif", "Normal", 16)}
    clipPlaneInfo = {}
    silhouettes = {0: True, 1: True, 2: True, 17: True}

    replyobj.status("Restoring window...", blankAfter=0, secondary=True)
    restoreWindowSize(windowSize)
    replyobj.status("Restoring open states...", blankAfter=0, secondary=True)
    restoreOpenStates(xformMap)
    replyobj.status("Restoring font info...", blankAfter=0, secondary=True)
    restoreFontInfo(fontInfo)
    replyobj.status("Restoring selections...", blankAfter=0, secondary=True)
    restoreSelections(curSelIds, savedSels)
    replyobj.status("Restoring openModel attributes...", blankAfter=0, secondary=True)
    restoreOpenModelsAttrs(openModelsAttrs)
    replyobj.status("Restoring model clipping...", blankAfter=0, secondary=True)
    restoreModelClip(clipPlaneInfo)
    replyobj.status("Restoring per-model silhouettes...", blankAfter=0, secondary=True)
    restoreSilhouettes(silhouettes)

    replyobj.status(
        "Restoring remaining extension info...", blankAfter=0, secondary=True
    )


try:
    restoreRemainder()
except:
    reportRestoreError("Error restoring post-model state")
from SimpleSession.versions.v65 import makeAfterModelsCBs

makeAfterModelsCBs()

from SimpleSession.versions.v65 import endRestore

replyobj.status("Finishing restore...", blankAfter=0, secondary=True)
endRestore({})
replyobj.status("", secondary=True)
replyobj.status("Restore finished.")
