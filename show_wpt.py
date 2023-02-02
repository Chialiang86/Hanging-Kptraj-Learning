import pybullet as p
import xml.etree.ElementTree as ET

# base_urdf_path = 'models/base/base.urdf'
# tree = ET.parse(base_urdf_path)
# root = tree.getroot()
# center = ''.join(str(i) + ' ' for i in mesh.centroid.tolist()).strip()
# root[0].find('inertial').find('origin').attrib['xyz'] = center

def main():
    base_urdf_path = 'shapes/waypoint/waypoint.urdf'
    tree = ET.parse(base_urdf_path)
    root = tree.getroot()

    # Create pybullet GUI
    physicsClientId = p.connect(p.GUI)
    # p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)
    p.resetDebugVisualizerCamera(
        cameraDistance=0.2,
        cameraYaw=90,
        cameraPitch=-10,
        cameraTargetPosition=[0.0, 0.0, 0.0]
    )
    p.resetSimulation()
    p.setPhysicsEngineParameter(numSolverIterations=150)
    sim_timestep = 1.0 / 240
    p.setTimeStep(sim_timestep)
    p.setGravity(0, 0, 0)

    oid = p.loadURDF(base_urdf_path)
    p.resetBasePositionAndOrientation(oid, [0, 0, 0], [0, 0, 0, 1])

    while True:
        keys = p.getKeyboardEvents()

        if ord('q') in keys and keys[ord('q')] & p.KEY_WAS_TRIGGERED:
            break

    return

if __name__=="__main__":
    main()