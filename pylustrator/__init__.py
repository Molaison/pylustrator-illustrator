#!/usr/bin/env python
# -*- coding: utf-8 -*-
# __init__.py

# Copyright (c) 2016-2020, Richard Gerum
#
# This file is part of Pylustrator.
#
# Pylustrator is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Pylustrator is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Pylustrator. If not, see <http://www.gnu.org/licenses/>

from .QtGuiDrag import initialize as start
from .helper_functions import (
    fig_text,
    add_axes,
    add_image,
    despine,
    changeFigureSize,
    mark_inset,
    VoronoiPlot,
    selectRectangle,
    mark_inset_pos,
    draw_from_point_to_bbox,
    draw_from_point_to_point,
    loadFigureFromFile,
    add_letter,
    add_letters,
)
from .QtGui import initialize as StartColorChooser
from .lab_colormap import LabColormap
from .helper_functions import loadFigureFromFile as load
from .artist_adapters import (
    AppearanceScalePlan,
    ArtistAdapter,
    ArtistAdapterRegistry,
    ArtistCapabilities,
    ChangeRecord,
    RigidRotationPlan,
    UnsupportedArtistError,
    get_artist_adapter,
    register_artist_adapter,
)
from .interaction import HitCandidate, HitStack, SelectionKernel, SelectionMode
from .legend_replay import UnsupportedLegendEntry, register_legend_entry_serializer
from .editor_model import EditorGroup, EditorScene
from .operations import OperationSupport, TransformIntent, TransformOperation
from .transform_engine import TransformPlan, TransformPreflightError
from .commands import (
    GENERATED_STATE_VERSION,
    ObjectLocator,
    migrate_generated_command,
    migrate_generated_source,
)
from .source_doctor import (
    SourceDiagnostic,
    SourceDoctorReport,
    diagnose_generated_source,
)

__version__ = "1.3.0"

__all__ = [
    "start",
    "fig_text",
    "add_axes",
    "add_image",
    "despine",
    "changeFigureSize",
    "mark_inset",
    "VoronoiPlot",
    "selectRectangle",
    "mark_inset_pos",
    "draw_from_point_to_bbox",
    "draw_from_point_to_point",
    "loadFigureFromFile",
    "add_letter",
    "add_letters",
    "StartColorChooser",
    "LabColormap",
    "load",
    "AppearanceScalePlan",
    "ArtistAdapter",
    "ArtistAdapterRegistry",
    "ArtistCapabilities",
    "ChangeRecord",
    "RigidRotationPlan",
    "UnsupportedArtistError",
    "get_artist_adapter",
    "register_artist_adapter",
    "UnsupportedLegendEntry",
    "register_legend_entry_serializer",
    "HitCandidate",
    "HitStack",
    "SelectionKernel",
    "SelectionMode",
    "EditorGroup",
    "EditorScene",
    "OperationSupport",
    "TransformIntent",
    "TransformOperation",
    "TransformPlan",
    "TransformPreflightError",
    "GENERATED_STATE_VERSION",
    "ObjectLocator",
    "migrate_generated_command",
    "migrate_generated_source",
    "SourceDiagnostic",
    "SourceDoctorReport",
    "diagnose_generated_source",
]
