from distutils.core import setup

setup(
    name='stompy',
    version='0.1',
    packages=['stompy', 'stompy.grid', 'stompy.io', 'stompy.io.local',
              'stompy.model', 'stompy.model.delft', 'stompy.model.fvcom',
              'stompy.model.pypart', 'stompy.model.suntans',
              'stompy.plot', 'stompy.plot.cmaps',
              'stompy.spatial'],
    package_data={'stompy':['tide_consts.txt']},
    license='MIT',
    url="https://github.com/rustychris/stompy",
    author="Rusty Holleman",
    author_email="rustychris@gmail.com",
    long_description=open('README.md').read(),
)
