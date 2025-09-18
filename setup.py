from setuptools import setup, find_packages

setup(
    name='sdforge',
    version='0.2.1',
    author='nassimberrada',
    author_email='your.email@example.com',
    description='A Python library for SDF modeling with real-time GLSL rendering and mesh export.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/sdforge',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'numpy',
        'scikit-image>=0.17',
        'watchdog',
        'moderngl',
        'glfw',
    ],
    extras_require={
        'gpu': [], # This is now empty, but kept for compatibility
        'record': [
            'imageio',
            'imageio-ffmpeg',
        ]
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Topic :: Multimedia :: Graphics :: 3D Modeling',
        'Topic :: Scientific/Engineering :: Visualization',
    ],
    python_requires='>=3.6',
)