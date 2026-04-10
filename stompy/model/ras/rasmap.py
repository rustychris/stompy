import os
import xml.etree.ElementTree as ET
from . import result_reader

class RasMap:
    """
    Read/write xml-formatted rasmap files
    """
    def __init__(self,rasmap_fn):
        self.fn = rasmap_fn
        self.tree = ET.parse(self.fn)
        self.root = self.tree.getroot()
        
    def summary(self):
        assert self.root.tag == 'RASMapper'

        return self.result_layers(self.root)

    @property
    def project_dir(self):
        return os.path.dirname(self.fn)
    
    def write(self,out_fn):
        # seems to round-trip just fine
        self.tree.write(out_fn)

    def add_result_layer(self,hdf_fn,name=None):
        if not hdf_fn.lower().endswith('.h5'):
            hdf_fn = hdf_fn + ".hdf"
        if not os.path.exists(self.absolute_path(hdf_fn)):
            raise Exception(f"Adding result layer: file not found {self.absolute_path(hdf_fn)}")
        if self.result_layer_exists(hdf_fn):
            raise Exception("Add result layer: layer already exists referencing same HDF")
        new_layer = self.create_result_layer(hdf_fn)
        results = self.root.find('./Results')
        results.append(new_layer)
        
    def result_layer_exists(self,hdf_fn):
        hdf_fn = self.normalize_path(hdf_fn)
        for layer in self.root.findall('./Results/Layer'):
            if layer.attrib['Filename'] == hdf_fn:
                return True
        return False
    
    def normalize_path(self, fn):
        """ normalize file to have .\ prefix if input is relative path.
        """
        if not os.path.isabs(fn) and not fn.startswith(".\\"):
            fn=".\\" + fn
        # Maybe should check for absolute path that could be relative?
        return fn
    
    def absolute_path(self,fn):
        if not os.path.isabs(fn):
            fn = os.path.join(self.project_dir,fn)
        return fn

    @staticmethod
    def result_layers(root):
        layers = root.findall('./Results/Layer')
        return "\n".join( [ f"{l.attrib['Name']}  {l.attrib['Filename']}"
                            for l in layers ] )
        
    def create_result_layer(self,hdf_fn):
        hdf_fn=self.normalize_path(hdf_fn)
        # This is the name displayed in the UI tree, and corresponds to plan short ID
        short_id = result_reader.RasReader.get_short_id(self.absolute_path(hdf_fn))
        print(f"Creating result layer, name = short_id = {short_id}")
        template = f'''
    <Layer Name="{short_id}" Type="RASResults" Filename="{hdf_fn}">
      <Layer Name="Event Conditions" Type="RASEventConditions" Filename="{hdf_fn}">
        <Layer Name="Wind Layer" Type="ResultWindLayer" Filename="{hdf_fn}">
          <ResampleMethod>near</ResampleMethod>
          <Surface On="True" />
          <Metadata BandIndex="0" SubDataset="" />
        </Layer>
        <Layer Name="Wind Layer" Type="ResultWindLayer" Filename="{hdf_fn}">
          <ResampleMethod>near</ResampleMethod>
          <Surface On="True" />
          <Metadata BandIndex="0" SubDataset="" />
        </Layer>
      </Layer>
      <Layer Type="RASGeometry" Checked="True" Filename="{hdf_fn}">
        <Layer Type="RASXS" UnitsRiverStation="Feet" RiverStationDecimalPlaces="0" />
        <Layer Name="Culvert Groups" Type="VirtualGeometryFeatureLayer" ParentIdentifiers="" />
        <Layer Name="Culvert Barrels" Type="VirtualGeometryFeatureLayer" ParentIdentifiers="Bridges/Culverts" />
        <Layer Name="Gate Groups" Type="VirtualGeometryFeatureLayer" ParentIdentifiers="" />
        <Layer Name="Gate Openings" Type="VirtualGeometryFeatureLayer" ParentIdentifiers="Inline Structures" />
        <Layer Name="Culvert Groups" Type="VirtualGeometryFeatureLayer" ParentIdentifiers="" />
        <Layer Name="Culvert Barrels" Type="VirtualGeometryFeatureLayer" ParentIdentifiers="Inline Structures" />
        <Layer Name="Rating Curve Outlets" Type="VirtualGeometryFeatureLayer" ParentIdentifiers="Inline Structures" />
        <Layer Name="Outlet Time Series" Type="VirtualGeometryFeatureLayer" ParentIdentifiers="Inline Structures" />
        <Layer Name="Gate Groups" Type="VirtualGeometryFeatureLayer" ParentIdentifiers="" />
        <Layer Name="Gate Openings" Type="VirtualGeometryFeatureLayer" ParentIdentifiers="Lateral Structures" />
        <Layer Name="Culvert Groups" Type="VirtualGeometryFeatureLayer" ParentIdentifiers="" />
        <Layer Name="Culvert Barrels" Type="VirtualGeometryFeatureLayer" ParentIdentifiers="Lateral Structures" />
        <Layer Name="Rating Curve Outlets" Type="VirtualGeometryFeatureLayer" ParentIdentifiers="Lateral Structures" />
        <Layer Name="Outlet Time Series" Type="VirtualGeometryFeatureLayer" ParentIdentifiers="Lateral Structures" />
        <Layer Name="Gate Groups" Type="VirtualGeometryFeatureLayer" ParentIdentifiers="" />
        <Layer Name="Gate Openings" Type="VirtualGeometryFeatureLayer" ParentIdentifiers="SA/2D Connections" />
        <Layer Name="Culvert Groups" Type="VirtualGeometryFeatureLayer" ParentIdentifiers="" />
        <Layer Name="Culvert Barrels" Type="VirtualGeometryFeatureLayer" ParentIdentifiers="SA/2D Connections" />
        <Layer Name="Rating Curve Outlets" Type="VirtualGeometryFeatureLayer" ParentIdentifiers="SA/2D Connections" />
        <Layer Name="Outlet Time Series" Type="VirtualGeometryFeatureLayer" ParentIdentifiers="SA/2D Connections" />
        <Layer Type="StructureLayer" UnitsRiverStation="Feet" RiverStationDecimalPlaces="0" />
        <Layer Type="RASPipeNodes">
          <SymbologyByAttributeColumns Checked="True" SelectedColumn="Node Type">
            <ValueToSymbology Value="Start">
              <Symbology>
                <PointSymbol Type="1" Size="8" pA="255" pR="255" pG="255" pB="255" bA="255" bR="0" bG="128" bB="0" />
                <HillShade On="True" Azimuth="315" Zenith="45" ZFactor="3" />
                <Contour On="False" Interval="5" Color="-16777216" />
                <Pen B="0" G="128" R="0" A="255" Dash="0" Width="2" />
                <SurfaceFill Colors="-1,-16777216" Values="0,100" Stretched="True" AlphaTag="255" UseDatasetMinMax="False" RegenerateForScreen="False" />
                <Brush Type="SolidBrush" B="0" G="128" R="0" A="50" Name="PolygonFill" />
              </Symbology>
            </ValueToSymbology>
            <ValueToSymbology Value="Junction">
              <Symbology>
                <PointSymbol Type="1" Size="8" pA="255" pR="255" pG="255" pB="255" bA="255" bR="128" bG="0" bB="0" />
                <HillShade On="True" Azimuth="315" Zenith="45" ZFactor="3" />
                <Contour On="False" Interval="5" Color="-16777216" />
                <Pen B="0" G="0" R="128" A="255" Dash="0" Width="2" />
                <SurfaceFill Colors="-1,-16777216" Values="0,100" Stretched="True" AlphaTag="255" UseDatasetMinMax="False" RegenerateForScreen="False" />
                <Brush Type="SolidBrush" B="0" G="0" R="128" A="50" Name="PolygonFill" />
              </Symbology>
            </ValueToSymbology>
            <ValueToSymbology Value="Culvert Opening">
              <Symbology>
                <PointSymbol Type="1" Size="8" pA="255" pR="128" pG="128" pB="128" bA="255" bR="128" bG="128" bB="128" />
                <HillShade On="True" Azimuth="315" Zenith="45" ZFactor="3" />
                <Contour On="False" Interval="5" Color="-16777216" />
                <Pen B="128" G="128" R="128" A="255" Dash="0" Width="2" />
                <SurfaceFill Colors="-1,-16777216" Values="0,100" Stretched="True" AlphaTag="255" UseDatasetMinMax="False" RegenerateForScreen="False" />
                <Brush Type="SolidBrush" B="128" G="128" R="128" A="50" Name="PolygonFill" />
              </Symbology>
            </ValueToSymbology>
            <ValueToSymbology Value="External">
              <Symbology>
                <PointSymbol Type="1" Size="8" pA="255" pR="255" pG="255" pB="0" bA="255" bR="255" bG="255" bB="0" />
                <HillShade On="True" Azimuth="315" Zenith="45" ZFactor="3" />
                <Contour On="False" Interval="5" Color="-16777216" />
                <Pen B="0" G="255" R="255" A="255" Dash="0" Width="2" />
                <SurfaceFill Colors="-1,-16777216" Values="0,100" Stretched="True" AlphaTag="255" UseDatasetMinMax="False" RegenerateForScreen="False" />
                <Brush Type="SolidBrush" B="0" G="255" R="255" A="50" Name="PolygonFill" />
              </Symbology>
            </ValueToSymbology>
            <ValueToSymbology Value="Closed">
              <Symbology>
                <PointSymbol Type="1" Size="8" pA="255" pR="255" pG="255" pB="255" bA="255" bR="0" bG="0" bB="255" />
                <HillShade On="True" Azimuth="315" Zenith="45" ZFactor="3" />
                <Contour On="False" Interval="5" Color="-16777216" />
                <Pen B="255" G="0" R="0" A="255" Dash="0" Width="2" />
                <SurfaceFill Colors="-1,-16777216" Values="0,100" Stretched="True" AlphaTag="255" UseDatasetMinMax="False" RegenerateForScreen="False" />
                <Brush Type="SolidBrush" B="255" G="0" R="0" A="50" Name="PolygonFill" />
              </Symbology>
            </ValueToSymbology>
            <ValueToSymbology Value="Error">
              <Symbology>
                <PointSymbol Type="1" Size="8" pA="255" pR="255" pG="0" pB="0" bA="255" bR="255" bG="0" bB="0" />
                <HillShade On="True" Azimuth="315" Zenith="45" ZFactor="3" />
                <Contour On="False" Interval="5" Color="-16777216" />
                <Pen B="0" G="0" R="255" A="255" Dash="0" Width="2" />
                <SurfaceFill Colors="-1,-16777216" Values="0,100" Stretched="True" AlphaTag="255" UseDatasetMinMax="False" RegenerateForScreen="False" />
                <Brush Type="SolidBrush" B="0" G="0" R="255" A="50" Name="PolygonFill" />
              </Symbology>
            </ValueToSymbology>
          </SymbologyByAttributeColumns>
        </Layer>
        <Layer Type="FinalNValueLayer" Checked="True">
          <ResampleMethod>near</ResampleMethod>
          <Surface On="True" />
        </Layer>
        <Layer Name="Final Values" Type="InterpretationRasterizerLayer" Filename="{hdf_fn}">
          <ResampleMethod>near</ResampleMethod>
          <Surface On="True" />
        </Layer>
        <Layer Name="Final Values" Type="InterpretationRasterizerLayer" Filename="{hdf_fn}">
          <ResampleMethod>near</ResampleMethod>
          <Surface On="True" />
        </Layer>
        <Layer Name="Manning's n" Type="InterpretationOverrideGroupLayer" Checked="True" />
      </Layer>
      <Layer Name="Plan" Type="RASPlan" Filename="{hdf_fn}" GeometryHDF="{hdf_fn}">
        <Layer Name="Encroachments" Type="RASEncroachments" Filename="{hdf_fn}" />
        <Layer Name="Zones" Type="RASEncroachmentZones" Filename="{hdf_fn}" />
        <Layer Name="Regions" Type="RASEncroachmentPolygons" Filename="{hdf_fn}" />
      </Layer>
      <Layer Name="Depth" Type="RASResultsMap">
        <MapParameters MapType="depth" ProfileIndex="2147483647" ProfileName="Max" />
      </Layer>
      <Layer Name="Velocity" Type="RASResultsMap" Checked="True">
        <MapParameters MapType="velocity" ProfileIndex="2572" ProfileName="12AUG2022 02:52:00" />
      </Layer>
      <Layer Name="WSE" Type="RASResultsMap">
        <MapParameters MapType="elevation" ProfileIndex="1780" ProfileName="11AUG2022 13:40:00" />
      </Layer>
    </Layer>
'''
        # ET.fromstring => Element (not a tree like ET.parse())
        return ET.fromstring(template)
