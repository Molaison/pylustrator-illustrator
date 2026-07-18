from __future__ import annotations

from typing import Optional

import qtawesome as qta
from qtpy import QtCore, QtGui, QtWidgets

import matplotlib as mpl
from matplotlib.artist import Artist

from pylustrator.drag_helper import get_artist_children
from pylustrator.editor_model import EditorGroup


class myTreeWidgetItem(QtGui.QStandardItem):
    def __init__(self, parent: QtWidgets.QWidget = None):
        """a tree view item to display the contents of the figure"""
        QtGui.QStandardItem.__init__(self, parent)

    def __lt__(self, otherItem: QtGui.QStandardItem):
        """how to sort the items"""
        if self.sort is None:
            return 0
        return self.sort < otherItem.sort


class MyTreeView(QtWidgets.QTreeView):
    # item_selected = lambda x, y: 0
    def item_clicked(x, y):
        return 0

    def item_activated(x, y):
        return 0

    def item_hoverEnter(x, y):
        return 0

    def item_hoverLeave(x, y):
        return 0

    last_selection = None
    last_hover = None

    def item_selected(self, x):
        if not getattr(self.fig, "no_figure_dragger_selection_update", False):
            if getattr(self.fig, "figure_dragger", None) is not None:
                self.fig.figure_dragger.select_element(x)

    def __init__(self, signals, layout: QtWidgets.QLayout):
        """A tree view to display the contents of a figure

        Args:
            parent: the parent widget
            layout: the layout to which to add the tree view
            fig: the target figure
        """
        super().__init__()
        # self.setMaximumWidth(300)

        signals.figure_changed.connect(self.setFigure)
        signals.figure_element_selected.connect(self.select_element)
        signals.figure_element_child_created.connect(
            lambda x: self.updateEntry(x, update_children=True)
        )

        layout.addWidget(self)

        # start a list for backwards search (from marker entry back to tree entry)
        self.marker_modelitems = {}
        self.marker_type_modelitems = {}

        # model for tree view
        self.model = QtGui.QStandardItemModel(0, 0)

        # some settings for the tree
        self.setUniformRowHeights(True)
        self.setHeaderHidden(True)
        self.setAnimated(True)
        self.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.setModel(self.model)
        self.expanded.connect(self.TreeExpand)
        self.clicked.connect(self.treeClicked)
        self.activated.connect(self.treeActivated)
        self.selectionModel().selectionChanged.connect(self.selectionChanged)

        # add context menu
        self.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)

        # add hover highlight
        self.viewport().setMouseTracking(True)
        self.viewport().installEventFilter(self)

        self.item_lookup = {}

    def select_element(self, element: Artist):
        """select an element"""
        selected_entries = []
        for index in self.selectionModel().selectedIndexes():
            if index.column() != 0:
                continue
            try:
                selected_entries.append(index.model().itemFromIndex(index).entry)
            except AttributeError:
                continue
        if element in selected_entries:
            return
        if element is None:
            self.setCurrentIndex(self.fig)
        else:
            self.setCurrentIndex(element)

    def selectionCommand(
        self, index: QtCore.QModelIndex, event: QtCore.QEvent = None
    ) -> QtCore.QItemSelectionModel.SelectionFlags:
        command = super().selectionCommand(index, event)
        modifiers = event.modifiers() if event is not None else QtCore.Qt.NoModifier
        additive_modifier = QtCore.Qt.ControlModifier | QtCore.Qt.MetaModifier
        if modifiers & additive_modifier:
            return QtCore.QItemSelectionModel.Toggle | QtCore.QItemSelectionModel.Rows
        return command | QtCore.QItemSelectionModel.Rows

    def setFigure(self, fig):
        self.fig = fig
        self.model.removeRows(0, self.model.rowCount())
        self.expand(None)

        self.deleteEntry(self.fig)
        self.expand(None)
        self.expand(self.fig)
        self.setCurrentIndex(self.fig)

    def selectionChanged(
        self, selection: QtCore.QItemSelection, y: QtCore.QItemSelection
    ):
        """when the selection in the tree view changes"""
        if getattr(self, "fig", None) is None:
            return
        entries = []
        for index in self.selectionModel().selectedIndexes():
            if index.column() != 0:
                continue
            try:
                entry = index.model().itemFromIndex(index).entry
            except AttributeError:
                continue
            if entry is not None and entry not in entries:
                entries.append(entry)
        entry = entries[-1] if entries else None
        current_index = self.currentIndex()
        if current_index.isValid():
            try:
                current_entry = current_index.model().itemFromIndex(current_index).entry
            except AttributeError:
                current_entry = None
            if current_entry in entries:
                entry = current_entry
        if self.last_selection != entries:
            self.last_selection = list(entries)
            if len(entries) > 1 and not getattr(
                self.fig, "no_figure_dragger_selection_update", False
            ):
                if getattr(self.fig, "figure_dragger", None) is not None:
                    self.fig.figure_dragger.select_elements(entries, primary=entry)
            else:
                self.item_selected(entry)

    def setCurrentIndex(self, entry: Artist):
        """set the currently selected entry"""
        while entry:
            item = self.getItemFromEntry(entry)
            if item is not None:
                try:
                    index = item.index()
                    self.selectionModel().select(
                        index,
                        QtCore.QItemSelectionModel.ClearAndSelect
                        | QtCore.QItemSelectionModel.Rows,
                    )
                    super().setCurrentIndex(index)
                except RuntimeError:  # maybe find out why we run into this error when the figure is changed
                    pass
                return
            try:
                entry = self.getParentEntry(entry)
            except (AttributeError, RuntimeError):
                return

    def treeClicked(self, index: QtCore.QModelIndex):
        """upon selecting one of the tree elements"""
        data = index.model().itemFromIndex(index).entry
        return self.item_clicked(data)

    def treeActivated(self, index: QtCore.QModelIndex):
        """upon selecting one of the tree elements"""
        data = index.model().itemFromIndex(index).entry
        return self.item_activated(data)

    def eventFilter(self, object: QtWidgets.QWidget, event: QtCore.QEvent):
        """event filter for tree view port to handle mouse over events and marker highlighting"""
        if event.type() == QtCore.QEvent.HoverMove:
            index = self.indexAt(event.pos())
            try:
                item = index.model().itemFromIndex(index)
                entry = item.entry
            except AttributeError:
                item = None
                entry = None

            # check for new item
            if entry != self.last_hover:
                # deactivate last hover item
                if self.last_hover is not None:
                    self.item_hoverLeave(self.last_hover)

                # activate current hover item
                if entry is not None:
                    self.item_hoverEnter(entry)

                self.last_hover = entry
                return True

        return False

    def queryToExpandEntry(self, entry: Artist) -> list:
        """when expanding a tree item"""
        if entry is None:
            return [self.fig]
        children = get_artist_children(entry)
        dragger = getattr(getattr(self, "fig", None), "figure_dragger", None)
        scene = getattr(dragger, "editor_scene", None)
        if scene is not None:
            return scene.tree_children(entry, children)
        return children

    def getParentEntry(self, entry: Artist) -> Artist:
        """get the parent of an item"""
        dragger = getattr(getattr(self, "fig", None), "figure_dragger", None)
        scene = getattr(dragger, "editor_scene", None)
        if scene is not None:
            return scene.tree_parent(entry)
        return getattr(entry, "tree_parent", None)

    def getNameOfEntry(self, entry: Artist) -> str:
        """convert an entry to a string"""
        if isinstance(entry, EditorGroup):
            return entry.name
        try:
            return str(entry)
        except AttributeError:
            return "unknown"

    def getIconOfEntry(self, entry: Artist) -> QtGui.QIcon:
        """get the icon of an entry"""
        dragger = getattr(getattr(self, "fig", None), "figure_dragger", None)
        scene = getattr(dragger, "editor_scene", None)
        if scene is not None:
            if scene.is_locked(entry):
                return qta.icon("fa5s.lock")
            if scene.is_explicitly_hidden(entry) or not entry.get_visible():
                return qta.icon("fa5s.eye-slash")
        if isinstance(entry, EditorGroup):
            return qta.icon("fa5s.layer-group")
        if getattr(entry, "_draggable", None):
            if entry._draggable.connected:
                return qta.icon("fa5.hand-paper-o")
        return QtGui.QIcon()

    def getEntrySortRole(self, entry: Artist):
        return None

    def getKey(self, entry: Artist) -> Artist:
        """get the key of an entry, which is the entry itself"""
        return entry

    def getItemFromEntry(self, entry: Artist) -> Optional[QtWidgets.QTreeWidgetItem]:
        """get the tree view item for the given artist"""
        if entry is None:
            return None
        key = self.getKey(entry)
        try:
            return self.item_lookup[key]
        except KeyError:
            return None

    def setItemForEntry(self, entry: Artist, item: QtWidgets.QTreeWidgetItem):
        """store a new artist and tree view widget pair"""
        key = self.getKey(entry)
        self.item_lookup[key] = item

    def expand(self, entry: Artist, force_reload: bool = True):
        """expand the children of a tree view item"""
        query = self.queryToExpandEntry(entry)
        parent_item = self.getItemFromEntry(entry)
        parent_entry = entry

        if parent_item:
            if parent_item.expanded is False:
                # remove the dummy child
                parent_item.removeRow(0)
                parent_item.expanded = True
            # force_reload: delete all child entries and re query content from DB
            elif force_reload:
                # delete child entries
                parent_item.removeRows(0, parent_item.rowCount())
            else:
                return

        # add all marker types
        row = -1
        for row, entry in enumerate(query):
            dragger = getattr(getattr(self, "fig", None), "figure_dragger", None)
            if getattr(dragger, "editor_scene", None) is None:
                entry.tree_parent = parent_entry
            if 1:
                if (
                    isinstance(entry, mpl.spines.Spine)
                    or isinstance(entry, mpl.axis.XAxis)
                    or isinstance(entry, mpl.axis.YAxis)
                ):
                    continue
                if isinstance(entry, mpl.text.Text) and entry.get_text() == "":
                    continue
                try:
                    if entry == parent_entry.patch:
                        continue
                except AttributeError:
                    pass
                try:
                    label = entry.get_label()
                    if label == "_tmp_snap" or label == "grabber":
                        continue
                except AttributeError:
                    pass
            self.addChild(parent_item, entry)

    def addChild(self, parent_item: QtWidgets.QWidget, entry: Artist, row=None):
        """add a child to a tree view node"""
        if parent_item is None:
            parent_item = self.model

        # add item
        item = myTreeWidgetItem(self.getNameOfEntry(entry))
        item.expanded = False
        item.entry = entry

        item.setIcon(self.getIconOfEntry(entry))
        item.setEditable(False)
        item.sort = self.getEntrySortRole(entry)

        if parent_item is None:
            if row is None:
                row = self.model.rowCount()
            self.model.insertRow(row)
            self.model.setItem(row, 0, item)
        else:
            if row is None:
                parent_item.appendRow(item)
            else:
                parent_item.insertRow(row, item)
        self.setItemForEntry(entry, item)

        # add dummy child
        if self.queryToExpandEntry(entry) is not None and len(
            self.queryToExpandEntry(entry)
        ):
            child = QtGui.QStandardItem("loading")
            child.entry = None
            child.setEditable(False)
            child.setIcon(qta.icon("fa5s.hourglass-half"))
            item.appendRow(child)
            item.expanded = False
        return item

    def TreeExpand(self, index):
        """expand a tree view node"""
        # Get item and entry
        item = index.model().itemFromIndex(index)
        entry = item.entry
        thread = None

        # Expand
        if item.expanded is False:
            self.expand(entry)
            # thread = Thread(target=self.expand, args=(entry,))

        # Start thread as daemonic
        if thread:
            thread.setDaemon(True)
            thread.start()

    def updateEntry(
        self,
        entry: Artist,
        update_children: bool = False,
        insert_before: Artist = None,
        insert_after: Artist = None,
    ):
        """update a tree view node"""
        # get the tree view item for the database entry
        item = self.getItemFromEntry(entry)
        # if we haven't one yet, we have to create it
        if item is None:
            # get the parent entry
            parent_entry = self.getParentEntry(entry)
            # if we have a parent and are not at the top level try to get the corresponding item
            if parent_entry:
                parent_item = self.getItemFromEntry(parent_entry)
                # parent item not in list or not expanded, than we don't need to update it because it is not shown
                if parent_item is None or parent_item.expanded is False:
                    if parent_item:
                        parent_item.setText(self.getNameOfEntry(parent_entry))
                    return
            else:
                parent_item = None

            # define the row where the new item should be
            row = None
            if insert_before:
                row = self.getItemFromEntry(insert_before).row()
            if insert_after:
                row = self.getItemFromEntry(insert_after).row() + 1

            # add the item as a child of its parent
            self.addChild(parent_item, entry, row)
            if parent_item:
                if row is None:
                    parent_item.sortChildren(0)
                if parent_entry:
                    parent_item.setText(self.getNameOfEntry(parent_entry))
        else:
            # check if we have to change the parent
            parent_entry = self.getParentEntry(entry)
            parent_item = self.getItemFromEntry(parent_entry)
            if parent_item != item.parent():
                # remove the item from the old position
                if item.parent() is None:
                    self.model.takeRow(item.row())
                else:
                    item.parent().takeRow(item.row())

                # determine a potential new position
                row = None
                if insert_before:
                    row = self.getItemFromEntry(insert_before).row()
                if insert_after:
                    row = self.getItemFromEntry(insert_after).row() + 1

                # move the item to the new position
                if parent_item is None:
                    if row is None:
                        row = self.model.rowCount()
                    self.model.insertRow(row)
                    self.model.setItem(row, 0, item)
                else:
                    if row is None:
                        parent_item.appendRow(item)
                    else:
                        parent_item.insertRow(row, item)

            # update the items name, icon and children
            item.setIcon(self.getIconOfEntry(entry))
            item.setText(self.getNameOfEntry(entry))
            if update_children:
                self.expand(entry, force_reload=True)

    def deleteEntry(self, entry: Artist):
        """delete an entry from the tree"""
        # get the tree view item for the database entry
        item = self.getItemFromEntry(entry)
        if item is None:
            return

        parent_item = item.parent()
        if parent_item:
            parent_entry = parent_item.entry

        key = self.getKey(entry)
        del self.item_lookup[key]

        # delete row from the treeview
        if parent_item is None:
            self.model.removeRow(item.row())
        else:
            item.parent().removeRow(item.row(), item.parent())

        # update the label of parent item
        if parent_item:
            name = self.getNameOfEntry(parent_entry)
            if name is not None:
                parent_item.setLabel(name)
