import * as THREE from 'three';

/**
 * Canonical 17-bone humanoid rig matching ARSA's `BoneId` enum.
 */
export enum BoneId {
  Pelvis = 0,
  LowerSpine = 1,
  Chest = 2,
  Neck = 3,
  Head = 4,
  LeftUpperArm = 5,
  LeftForearm = 6,
  LeftHand = 7,
  RightUpperArm = 8,
  RightForearm = 9,
  RightHand = 10,
  LeftThigh = 11,
  LeftShin = 12,
  LeftFoot = 13,
  RightThigh = 14,
  RightShin = 15,
  RightFoot = 16,
}

export const BONE_COUNT = 17;

/** Physics collider shape type emitted by `get_body_descriptors()`. */
export const enum ShapeType {
  Capsule = 0,
  Box = 1,
  Sphere = 2,
}

/**
 * Shared test-dummy PBR material palette (Aperture / Euphoria test-chamber look).
 */
interface MannequinPalette {
  ceramic: THREE.MeshStandardMaterial;
  gunmetal: THREE.MeshStandardMaterial;
  amber: THREE.MeshStandardMaterial;
  visor: THREE.MeshStandardMaterial;
}

let sharedPalette: MannequinPalette | null = null;

function getPalette(): MannequinPalette {
  if (sharedPalette) return sharedPalette;

  sharedPalette = {
    ceramic: new THREE.MeshStandardMaterial({
      color: 0xd8dce2,
      roughness: 0.42,
      metalness: 0.06,
    }),
    gunmetal: new THREE.MeshStandardMaterial({
      color: 0x2a303a,
      roughness: 0.3,
      metalness: 0.72,
    }),
    amber: new THREE.MeshStandardMaterial({
      color: 0xe8942c,
      roughness: 0.35,
      metalness: 0.16,
      emissive: 0x3b1f04,
      emissiveIntensity: 0.25,
    }),
    visor: new THREE.MeshStandardMaterial({
      color: 0x141820,
      roughness: 0.16,
      metalness: 0.88,
    }),
  };
  return sharedPalette;
}

function enableShadows(root: THREE.Object3D): void {
  root.traverse((obj) => {
    if (obj instanceof THREE.Mesh) {
      obj.castShadow = true;
      obj.receiveShadow = true;
    }
  });
}

/**
 * Creates a smooth sculpted lathe segment centered along local Y (`-halfLen .. +halfLen`).
 */
function createLathedLimbGeo(
  profile: Array<[number, number]>,
  radialSegments = 20
): THREE.LatheGeometry {
  const pts = profile.map(([r, y]) => new THREE.Vector2(Math.max(0.001, r), y));
  return new THREE.LatheGeometry(pts, radialSegments);
}

/**
 * Creates a beveled extruded mesh centered at `(0, 0, 0)`.
 */
function createBeveledShapeGeo(
  shape: THREE.Shape,
  depth: number,
  bevelThickness: number,
  bevelSize: number,
  bevelSegments = 3
): THREE.ExtrudeGeometry {
  const geo = new THREE.ExtrudeGeometry(shape, {
    depth: Math.max(0.01, depth - bevelThickness * 2),
    bevelEnabled: true,
    bevelThickness,
    bevelSize,
    bevelSegments,
    curveSegments: 12,
  });
  geo.center();
  return geo;
}

/**
 * Builds an anatomical limb segment (UpperArm, Forearm, Thigh, Shin) with
 * tapered muscle contour, mechanical collar rings, and gunmetal joint spheres.
 */
function buildSculptedLimb(
  radius: number,
  halfLen: number,
  options: {
    topScale: number;
    bellyScale: number;
    bellyOffsetRatio: number;
    bottomScale: number;
    posteriorSwellZ?: number;
    amberPip?: boolean;
    sideSign?: number;
  }
): THREE.Group {
  const pal = getPalette();
  const group = new THREE.Group();

  const jointR = radius * 0.88;
  const sleeveHalf = Math.max(0.02, halfLen * 0.92);

  // Tapered ceramic polymer outer shell
  const profile: Array<[number, number]> = [
    [radius * 0.45 * options.bottomScale, -sleeveHalf],
    [radius * options.bottomScale, -sleeveHalf * 0.84],
    [radius * ((options.bottomScale + options.bellyScale) * 0.5), -sleeveHalf * 0.28],
    [radius * options.bellyScale, sleeveHalf * options.bellyOffsetRatio],
    [radius * ((options.topScale + options.bellyScale) * 0.52), sleeveHalf * 0.62],
    [radius * options.topScale, sleeveHalf * 0.86],
    [radius * 0.45 * options.topScale, sleeveHalf],
  ];

  const shellGeo = createLathedLimbGeo(profile, 22);
  if (options.posteriorSwellZ) {
    // Subtle anatomical depth bias (e.g., calf gastrocnemius toward -Z)
    const pos = shellGeo.attributes.position;
    for (let i = 0; i < pos.count; i++) {
      const y = pos.getY(i);
      const z = pos.getZ(i);
      if (z < 0) {
        const yNorm = 1 - Math.min(1, Math.abs(y / sleeveHalf - 0.15));
        pos.setZ(i, z * (1 + options.posteriorSwellZ * yNorm));
      }
    }
    shellGeo.computeVertexNormals();
  }

  const shellMesh = new THREE.Mesh(shellGeo, pal.ceramic);
  group.add(shellMesh);

  // Dark anodized aluminum joint spheres at proximal (+Y) and distal (-Y) ends
  const capGeo = new THREE.SphereGeometry(jointR, 18, 14);
  const topCap = new THREE.Mesh(capGeo, pal.gunmetal);
  topCap.position.y = halfLen;
  group.add(topCap);

  const botCap = new THREE.Mesh(capGeo, pal.gunmetal);
  botCap.position.y = -halfLen;
  group.add(botCap);

  // Mechanical collar rings separating joint caps from ceramic shell
  const collarTop = new THREE.Mesh(
    new THREE.TorusGeometry(radius * options.topScale * 0.92, radius * 0.09, 10, 22),
    pal.gunmetal
  );
  collarTop.rotation.x = Math.PI / 2;
  collarTop.position.y = sleeveHalf * 0.88;
  group.add(collarTop);

  const collarBot = new THREE.Mesh(
    new THREE.TorusGeometry(radius * options.bottomScale * 0.92, radius * 0.09, 10, 22),
    pal.gunmetal
  );
  collarBot.rotation.x = Math.PI / 2;
  collarBot.position.y = -sleeveHalf * 0.88;
  group.add(collarBot);

  if (options.amberPip) {
    const sign = options.sideSign ?? 1;
    const pip = new THREE.Mesh(
      new THREE.BoxGeometry(radius * 0.18, sleeveHalf * 0.42, radius * 0.36),
      pal.amber
    );
    pip.position.set(sign * radius * options.bellyScale * 0.94, sleeveHalf * 0.1, 0);
    group.add(pip);
  }

  return group;
}

/**
 * Sculpted iliac/pelvis assembly with hip socket cups and amber front chevron.
 */
function buildPelvis(dim0: number, dim1: number, dim2: number, shapeType: number): THREE.Group {
  const pal = getPalette();
  const group = new THREE.Group();

  const width = shapeType === ShapeType.Box ? dim0 * 2 : Math.max(0.24, dim0 * 2.35);
  const height = shapeType === ShapeType.Box ? dim1 * 2 : Math.max(0.16, (dim1 + dim0) * 1.45);
  const depth = shapeType === ShapeType.Box ? dim2 * 2 : Math.max(0.16, dim0 * 1.65);

  const hw = width * 0.5;
  const hh = height * 0.5;

  // Iliac crest beveled silhouette in XY plane, extruded along Z
  const iliac = new THREE.Shape();
  iliac.moveTo(-hw * 0.65, -hh);
  iliac.lineTo(hw * 0.65, -hh);
  iliac.lineTo(hw, -hh * 0.12);
  iliac.lineTo(hw * 0.86, hh);
  iliac.lineTo(-hw * 0.86, hh);
  iliac.lineTo(-hw, -hh * 0.12);
  iliac.closePath();

  const pelvisGeo = createBeveledShapeGeo(iliac, depth, depth * 0.14, width * 0.045, 3);
  group.add(new THREE.Mesh(pelvisGeo, pal.ceramic));

  // Central gunmetal pelvic spine core
  const core = new THREE.Mesh(
    new THREE.CylinderGeometry(depth * 0.36, depth * 0.32, height * 1.04, 16),
    pal.gunmetal
  );
  group.add(core);

  // Hip joint ball-socket housings
  const socketGeo = new THREE.SphereGeometry(height * 0.36, 14, 12);
  for (const side of [-1, 1]) {
    const socket = new THREE.Mesh(socketGeo, pal.gunmetal);
    socket.position.set(side * hw * 0.78, -hh * 0.38, 0);
    group.add(socket);
  }

  // Safety-amber pelvic belt stripe (+Z front)
  const stripe = new THREE.Mesh(
    new THREE.BoxGeometry(width * 0.56, height * 0.2, depth * 0.12),
    pal.amber
  );
  stripe.position.set(0, hh * 0.36, depth * 0.48);
  group.add(stripe);

  return group;
}

/**
 * Sculpted articulated lumbar spine segment with segmented gunmetal vertebrae.
 */
function buildLowerSpine(radius: number, halfLen: number): THREE.Group {
  const pal = getPalette();
  const group = new THREE.Group();

  const h = Math.max(0.1, halfLen * 2);
  const r = Math.max(0.06, radius);

  // Inner gunmetal column
  const core = new THREE.Mesh(
    new THREE.CylinderGeometry(r * 0.68, r * 0.64, h * 1.08, 16),
    pal.gunmetal
  );
  group.add(core);

  // Three stacked vertebra flanges
  for (const frac of [-0.3, 0, 0.3]) {
    const ring = new THREE.Mesh(
      new THREE.CylinderGeometry(r * 0.88, r * 0.88, h * 0.16, 18),
      pal.gunmetal
    );
    ring.scale.set(1.12, 1, 0.88);
    ring.position.y = frac * h;
    group.add(ring);
  }

  // Ceramic abdominal & dorsal protective shells
  const torsoShape = new THREE.Shape();
  const w = r * 1.05;
  const hh = h * 0.44;
  torsoShape.moveTo(-w * 0.85, -hh);
  torsoShape.lineTo(w * 0.85, -hh);
  torsoShape.lineTo(w, hh);
  torsoShape.lineTo(-w, hh);
  torsoShape.closePath();

  const shell = new THREE.Mesh(
    createBeveledShapeGeo(torsoShape, r * 1.48, r * 0.12, r * 0.08, 2),
    pal.ceramic
  );
  group.add(shell);

  return group;
}

/**
 * Tapered anatomical ribcage/chest with broad shoulder yoke, sternum contour,
 * gunmetal shoulder sockets, and safety-amber chest stripe.
 */
function buildChest(dim0: number, dim1: number, dim2: number, shapeType: number): THREE.Group {
  const pal = getPalette();
  const group = new THREE.Group();

  const shoulderHalfW =
    shapeType === ShapeType.Box ? dim0 : Math.max(0.165, dim0 * 1.38);
  const waistHalfW = shoulderHalfW * 0.72;
  const halfH =
    shapeType === ShapeType.Box ? dim1 : Math.max(0.13, dim1 + dim0 * 0.55);
  const depth =
    shapeType === ShapeType.Box ? dim2 * 2 : Math.max(0.18, dim0 * 1.52);

  // Tapered ribcage profile in XY
  const rib = new THREE.Shape();
  rib.moveTo(-waistHalfW, -halfH);
  rib.lineTo(waistHalfW, -halfH);
  rib.lineTo(shoulderHalfW, halfH * 0.42);
  rib.lineTo(shoulderHalfW * 0.78, halfH);
  rib.lineTo(-shoulderHalfW * 0.78, halfH);
  rib.lineTo(-shoulderHalfW, halfH * 0.42);
  rib.closePath();

  const ribGeo = createBeveledShapeGeo(rib, depth, depth * 0.16, shoulderHalfW * 0.09, 3);
  group.add(new THREE.Mesh(ribGeo, pal.ceramic));

  // Forward sternum pectoral plate (+Z) so front silhouette is unmistakable
  const pectShape = new THREE.Shape();
  const pw = shoulderHalfW * 0.72;
  const ph = halfH * 0.46;
  pectShape.moveTo(-pw * 0.75, -ph);
  pectShape.lineTo(pw * 0.75, -ph);
  pectShape.lineTo(pw, ph * 0.45);
  pectShape.lineTo(pw * 0.58, ph);
  pectShape.lineTo(-pw * 0.58, ph);
  pectShape.lineTo(-pw, ph * 0.45);
  pectShape.closePath();

  const pectMesh = new THREE.Mesh(
    createBeveledShapeGeo(pectShape, depth * 0.24, depth * 0.04, pw * 0.06, 2),
    pal.ceramic
  );
  pectMesh.position.set(0, halfH * 0.2, depth * 0.42);
  group.add(pectMesh);

  // Gunmetal spine dorsal ridge (-Z)
  const spineBlock = new THREE.Mesh(
    new THREE.BoxGeometry(shoulderHalfW * 0.34, halfH * 1.72, depth * 0.26),
    pal.gunmetal
  );
  spineBlock.position.set(0, 0, -depth * 0.42);
  group.add(spineBlock);

  // Gunmetal shoulder socket housings
  const socketGeo = new THREE.CylinderGeometry(
    depth * 0.28,
    depth * 0.28,
    shoulderHalfW * 0.26,
    14
  );
  for (const side of [-1, 1]) {
    const socket = new THREE.Mesh(socketGeo, pal.gunmetal);
    socket.rotation.z = Math.PI / 2;
    socket.position.set(side * shoulderHalfW * 0.96, halfH * 0.52, 0);
    group.add(socket);
  }

  // Safety-amber chest chevron / telemetry bar (+Z)
  const chestStripe = new THREE.Mesh(
    new THREE.BoxGeometry(shoulderHalfW * 0.92, halfH * 0.14, depth * 0.09),
    pal.amber
  );
  chestStripe.position.set(0, halfH * 0.28, depth * 0.52);
  group.add(chestStripe);

  const sternumPip = new THREE.Mesh(
    new THREE.BoxGeometry(shoulderHalfW * 0.12, halfH * 0.34, depth * 0.09),
    pal.amber
  );
  sternumPip.position.set(0, -halfH * 0.05, depth * 0.51);
  group.add(sternumPip);

  return group;
}

/**
 * Sleek mechanical neck column with cervical collar.
 */
function buildNeck(radius: number, halfLen: number): THREE.Group {
  const pal = getPalette();
  const group = new THREE.Group();

  const r = Math.max(0.036, radius * 0.9);
  const h = Math.max(0.06, halfLen * 2 + r * 0.6);

  const column = new THREE.Mesh(
    new THREE.CylinderGeometry(r * 0.84, r, h, 16),
    pal.gunmetal
  );
  group.add(column);

  const collar = new THREE.Mesh(
    new THREE.TorusGeometry(r * 1.08, r * 0.16, 10, 20),
    pal.ceramic
  );
  collar.rotation.x = Math.PI / 2;
  collar.position.y = -h * 0.26;
  group.add(collar);

  return group;
}

/**
 * Sculpted cranial oval + jaw/chin guard + recessed dark visor (+Z) + amber brow stripe.
 */
function buildHead(radius: number): THREE.Group {
  const pal = getPalette();
  const group = new THREE.Group();

  const r = Math.max(0.095, radius);

  // Cranial dome (slightly elongated vertically & front-to-back)
  const cranium = new THREE.Mesh(new THREE.SphereGeometry(r, 24, 18), pal.ceramic);
  cranium.scale.set(0.88, 1.12, 1.04);
  cranium.position.y = r * 0.12;
  group.add(cranium);

  // Contoured jaw / chin block facing +Z
  const jawShape = new THREE.Shape();
  jawShape.moveTo(-r * 0.58, -r * 0.45);
  jawShape.lineTo(r * 0.58, -r * 0.45);
  jawShape.lineTo(r * 0.72, r * 0.12);
  jawShape.lineTo(-r * 0.72, r * 0.12);
  jawShape.closePath();

  const jaw = new THREE.Mesh(
    createBeveledShapeGeo(jawShape, r * 1.12, r * 0.14, r * 0.1, 2),
    pal.ceramic
  );
  jaw.position.set(0, -r * 0.36, r * 0.22);
  group.add(jaw);

  // Dark optical visor band wrap (+Z front)
  const visor = new THREE.Mesh(
    new THREE.CylinderGeometry(r * 0.93, r * 0.9, r * 0.42, 20, 1, false, -Math.PI * 0.38, Math.PI * 0.76),
    pal.visor
  );
  visor.position.set(0, r * 0.16, r * 0.15);
  group.add(visor);

  // Subtle safety-amber brow accent stripe above visor
  const browStripe = new THREE.Mesh(
    new THREE.CylinderGeometry(r * 0.94, r * 0.94, r * 0.09, 20, 1, false, -Math.PI * 0.3, Math.PI * 0.6),
    pal.amber
  );
  browStripe.position.set(0, r * 0.44, r * 0.15);
  group.add(browStripe);

  // Gunmetal ear-pod sensors on left/right sides
  const earGeo = new THREE.CylinderGeometry(r * 0.24, r * 0.24, r * 0.16, 14);
  for (const side of [-1, 1]) {
    const ear = new THREE.Mesh(earGeo, pal.gunmetal);
    ear.rotation.z = Math.PI / 2;
    ear.position.set(side * r * 0.88, r * 0.06, -r * 0.04);
    group.add(ear);
  }

  return group;
}

/**
 * Sculpted palm/mitt hand with subtle thumb silhouette and gunmetal wrist cuff.
 */
function buildHand(dim0: number, dim1: number, dim2: number, shapeType: number, isLeft: boolean): THREE.Group {
  const pal = getPalette();
  const group = new THREE.Group();

  const w = shapeType === ShapeType.Box ? dim0 * 2 : Math.max(0.055, dim0 * 1.5);
  const h = shapeType === ShapeType.Box ? dim1 * 2 : Math.max(0.11, (dim1 + dim0) * 1.8);
  const d = shapeType === ShapeType.Box ? dim2 * 2 : Math.max(0.04, dim0 * 1.1);

  // Palm mitt profile
  const palm = new THREE.Shape();
  const hw = w * 0.5;
  const hh = h * 0.5;
  palm.moveTo(-hw * 0.85, -hh);
  palm.lineTo(hw * 0.85, -hh);
  palm.lineTo(hw, hh * 0.35);
  palm.lineTo(hw * 0.72, hh);
  palm.lineTo(-hw * 0.72, hh);
  palm.lineTo(-hw, hh * 0.35);
  palm.closePath();

  const palmMesh = new THREE.Mesh(
    createBeveledShapeGeo(palm, d, d * 0.18, hw * 0.12, 2),
    pal.ceramic
  );
  group.add(palmMesh);

  // Gunmetal wrist cuff
  const cuff = new THREE.Mesh(
    new THREE.CylinderGeometry(hw * 0.78, hw * 0.72, h * 0.22, 12),
    pal.gunmetal
  );
  cuff.position.y = hh * 0.85;
  group.add(cuff);

  // Angled thumb guard pointing inward/forward
  const thumbSide = isLeft ? -1 : 1;
  const thumb = new THREE.Mesh(
    new THREE.CapsuleGeometry(d * 0.28, h * 0.28, 4, 8),
    pal.ceramic
  );
  thumb.position.set(thumbSide * hw * 0.88, 0, d * 0.28);
  thumb.rotation.z = thumbSide * -0.45;
  thumb.rotation.x = 0.25;
  group.add(thumb);

  return group;
}

/**
 * Contoured athletic/lab shoe-foot block pointing along +Z.
 */
function buildFoot(dim0: number, dim1: number, dim2: number, shapeType: number): THREE.Group {
  const pal = getPalette();
  const group = new THREE.Group();

  const w = shapeType === ShapeType.Box ? dim0 * 2 : Math.max(0.085, dim0 * 1.9);
  const h = shapeType === ShapeType.Box ? dim1 * 2 : Math.max(0.075, dim0 * 1.7);
  const l = shapeType === ShapeType.Box ? dim2 * 2 : Math.max(0.21, (dim1 + dim0) * 2.2);

  // Side-profile shoe silhouette in ZY plane, extruded along X
  const sideProfile = new THREE.Shape();
  const hl = l * 0.5;
  const hh = h * 0.5;
  sideProfile.moveTo(-hl, -hh * 0.8);
  sideProfile.lineTo(hl * 0.85, -hh * 0.8);
  sideProfile.lineTo(hl, -hh * 0.15);
  sideProfile.lineTo(hl * 0.52, hh * 0.35);
  sideProfile.lineTo(-hl * 0.22, hh);
  sideProfile.lineTo(-hl * 0.92, hh * 0.88);
  sideProfile.lineTo(-hl, -hh * 0.2);
  sideProfile.closePath();

  const upperGeo = createBeveledShapeGeo(sideProfile, w, w * 0.14, h * 0.08, 2);
  // ExtrudeGeometry extrudes along Z; rotate so sideProfile's X maps to +Z forward
  upperGeo.rotateY(-Math.PI / 2);
  const upperMesh = new THREE.Mesh(upperGeo, pal.ceramic);
  group.add(upperMesh);

  // Dark gunmetal sole tread
  const sole = new THREE.Mesh(
    new THREE.BoxGeometry(w * 0.96, h * 0.24, l * 0.98),
    pal.gunmetal
  );
  sole.position.y = -hh * 0.78;
  group.add(sole);

  // Ankle collar socket
  const collar = new THREE.Mesh(
    new THREE.CylinderGeometry(w * 0.38, w * 0.42, h * 0.36, 14),
    pal.gunmetal
  );
  collar.position.set(0, hh * 0.85, -hl * 0.45);
  group.add(collar);

  return group;
}

/**
 * Builds an anatomically sculpted Three.js mesh group for the given `BoneId`,
 * scaled cleanly to match the physics collider descriptor `(shapeType, dim0, dim1, dim2)`.
 */
export function createBoneMesh(
  boneId: number,
  shapeType: number,
  dim0: number,
  dim1: number,
  dim2: number
): THREE.Group {
  let meshGroup: THREE.Group;

  const r = Math.max(0.02, dim0);
  const halfLen = Math.max(0.02, dim1);

  switch (boneId) {
    case BoneId.Pelvis:
      meshGroup = buildPelvis(dim0, dim1, dim2, shapeType);
      break;
    case BoneId.LowerSpine:
      meshGroup = buildLowerSpine(r, halfLen);
      break;
    case BoneId.Chest:
      meshGroup = buildChest(dim0, dim1, dim2, shapeType);
      break;
    case BoneId.Neck:
      meshGroup = buildNeck(r, halfLen);
      break;
    case BoneId.Head:
      meshGroup = buildHead(r);
      break;
    case BoneId.LeftUpperArm:
    case BoneId.RightUpperArm:
      meshGroup = buildSculptedLimb(r, halfLen, {
        topScale: 1.18,
        bellyScale: 1.06,
        bellyOffsetRatio: 0.12,
        bottomScale: 0.82,
        amberPip: true,
        sideSign: boneId === BoneId.LeftUpperArm ? 1 : -1,
      });
      break;
    case BoneId.LeftForearm:
    case BoneId.RightForearm:
      meshGroup = buildSculptedLimb(r, halfLen, {
        topScale: 1.12,
        bellyScale: 1.05,
        bellyOffsetRatio: 0.25,
        bottomScale: 0.72,
      });
      break;
    case BoneId.LeftHand:
    case BoneId.RightHand:
      meshGroup = buildHand(dim0, dim1, dim2, shapeType, boneId === BoneId.LeftHand);
      break;
    case BoneId.LeftThigh:
    case BoneId.RightThigh:
      meshGroup = buildSculptedLimb(r, halfLen, {
        topScale: 1.24,
        bellyScale: 1.12,
        bellyOffsetRatio: 0.14,
        bottomScale: 0.82,
        amberPip: true,
        sideSign: boneId === BoneId.LeftThigh ? 1 : -1,
      });
      break;
    case BoneId.LeftShin:
    case BoneId.RightShin:
      meshGroup = buildSculptedLimb(r, halfLen, {
        topScale: 1.06,
        bellyScale: 1.12,
        bellyOffsetRatio: 0.24,
        bottomScale: 0.68,
        posteriorSwellZ: 0.28,
      });
      break;
    case BoneId.LeftFoot:
    case BoneId.RightFoot:
      meshGroup = buildFoot(dim0, dim1, dim2, shapeType);
      break;
    default:
      meshGroup = buildSculptedLimb(r, halfLen, {
        topScale: 1.0,
        bellyScale: 1.0,
        bellyOffsetRatio: 0,
        bottomScale: 0.9,
      });
      break;
  }

  enableShadows(meshGroup);
  return meshGroup;
}
